"""End-to-end verification against a synthetic world with known ground truth.

Run with:  python -m tests.test_pipeline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


import dataset, schema  # noqa: E402
from dashboard import render  # noqa: E402
from features import FEATURE_COLUMNS, build_features, training_matrix  # noqa: E402
from model import CFBModel, evaluate, summarize, walk_forward  # noqa: E402
from ratings import PreseasonPriors, RatingsEngine  # noqa: E402
from simulate import FakeWeather, build_world  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    results.append((bool(condition), label))
    print(f"{PASS if condition else FAIL}  {label}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    print(f"Simulating seasons {years[0]}-{years[-1]}...")
    world = build_world(years)

    # ---------------------------------------------------------------- schema
    section("1. Schema normalisation")
    raw_games = pd.json_normalize(world["games"])
    raw_venues = pd.json_normalize(world["venues"])

    missing_g = [f for f in schema.missing_report(raw_games, schema.GAME_FIELDS)
                 if f in schema.REQUIRED_GAME_FIELDS]
    check(not missing_g, "all required game fields resolved",
          f"missing: {missing_g}" if missing_g else "")
    missing_v = [f for f in schema.missing_report(raw_venues, schema.VENUE_FIELDS)
                 if f in schema.REQUIRED_VENUE_FIELDS]
    check(not missing_v, "all required venue fields resolved",
          f"missing: {missing_v}" if missing_v else "")

    # snake_case variant should map just as well
    snake = raw_games.rename(columns={
        "homeTeam": "home_team", "awayTeam": "away_team",
        "homePoints": "home_points", "awayPoints": "away_points",
        "neutralSite": "neutral_site", "startDate": "start_date",
        "venueId": "venue_id", "seasonType": "season_type"})
    snake_missing = [f for f in schema.missing_report(snake, schema.GAME_FIELDS)
                     if f in schema.REQUIRED_GAME_FIELDS]
    check(not snake_missing, "snake_case API variant also resolves",
          f"missing: {snake_missing}" if snake_missing else "")

    # ---------------------------------------------------------------- clean
    section("2. Game table assembly")
    v = schema.normalise(raw_venues, schema.VENUE_FIELDS)
    for col in ("lat", "lon", "elevation", "capacity"):
        v[col] = pd.to_numeric(v[col], errors="coerce")
    v["dome"] = v["dome"].fillna(False).astype(bool)
    v["venue_id"] = pd.to_numeric(v["venue_id"], errors="coerce")

    g = schema.normalise(raw_games, schema.GAME_FIELDS)
    games = dataset.clean_games(g, v)
    check(len(games) == len(world["games"]), "no games lost in cleaning",
          f"{len(games)} of {len(world['games'])}")
    check(games["lat"].notna().all(), "every game has venue coordinates")
    check(games["kickoff"].notna().all(), "every kickoff parsed")

    games = dataset.add_rest_and_travel(games)
    check(games["away_travel_mi"].notna().all(), "travel distance computed")
    check(games["home_rest_days"].between(0, 30).all(), "rest days in range")

    line_rows = []
    for entry in world["lines"]:
        best = dataset._best_line(entry["lines"])
        line_rows.append({"game_id": entry["id"], **best})
    line_df = pd.DataFrame(line_rows)
    games = games.merge(line_df, on="game_id", how="left")
    check(games["spread"].notna().all(), "betting lines joined")

    # ---------------------------------------------------------------- leakage
    section("3. No-leakage guarantee")
    priors = PreseasonPriors()
    seed = RatingsEngine(games, PreseasonPriors())
    for year in years:
        prior_final = seed.final_ratings(year - 1) if year - 1 in years else None
        priors.build(year, None, None, None, prior_final)
    engine = RatingsEngine(games, priors)

    full = engine.ratings_before(2022, 8)

    truncated_games = games[~((games["season"] == 2022) & (games["week"] >= 8))].copy()
    engine_trunc = RatingsEngine(truncated_games, priors)
    trunc = engine_trunc.ratings_before(2022, 8)

    merged = full.merge(trunc, on="team", suffixes=("_full", "_trunc"))
    max_diff = float((merged["rating_full"] - merged["rating_trunc"]).abs().max())
    check(max_diff < 1e-9,
          "week-8 ratings identical with weeks 8+ deleted", f"max diff {max_diff:.2e}")

    wk1 = engine.ratings_before(2022, 1)
    check(float(wk1["played"].sum()) == 0.0,
          "week-1 ratings use zero played games")

    # ratings should correlate with the hidden truth by mid-season
    late = engine.ratings_before(2022, 12).set_index("team")["rating"]
    truth = pd.Series({t: s for t, s in zip(world["teams"]["team"],
                                            world["teams"]["latent"])})
    common = late.index.intersection(truth.index)
    corr = float(np.corrcoef(late[common], truth[common])[0, 1])
    check(corr > 0.75, "mid-season ratings track hidden team strength",
          f"r = {corr:.3f}")

    # ---------------------------------------------------------------- features
    section("4. Feature construction")
    weather = FakeWeather(world["games"], world["venues"])
    completed = games[games["completed"]].copy()
    feat = build_features(completed, engine, weather,
                          with_weather=True, historical=True)

    check(len(feat) == len(completed), "one feature row per game")
    missing_cols = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    check(not missing_cols, "all declared features present", str(missing_cols))
    nan_share = feat[FEATURE_COLUMNS].isna().mean().max()
    check(nan_share < 0.02, "features essentially free of NaNs",
          f"worst column {nan_share:.3%}")
    check(feat["temp_f"].std() > 5, "weather varies across games",
          f"temp sd {feat['temp_f'].std():.1f}F")

    # A failed weather lookup must leave NaN, not a plausible-looking constant.
    # Filling it in taught an earlier version that every unfetchable game was a
    # calm 70F afternoon, which is worse than admitting ignorance - the booster
    # handles NaN natively.
    import weather as weather_mod

    class _AlwaysFails:
        def prefetch(self, games, historical=True):
            pass

        def at_kickoff(self, *a, **k):
            return dict(weather_mod.UNKNOWN_CONDITIONS)

    blank = build_features(completed.head(40), engine, _AlwaysFails(),
                           with_weather=True, historical=True)
    check(blank["temp_f"].isna().all() and blank["wind_mph"].isna().all(),
          "unavailable weather stays NaN instead of being filled in")
    check(weather_mod.describe(dict(weather_mod.UNKNOWN_CONDITIONS))
          == "Weather unavailable",
          "dashboard admits when weather is missing")

    X, y = training_matrix(feat)
    check("spread" not in X.columns and "over_under" not in X.columns,
          "betting line excluded from model inputs")
    check(len(X) > 3000, "enough training rows", f"{len(X):,}")

    # ---------------------------------------------------------------- model
    section("5. Model training and walk-forward evaluation")
    oos = walk_forward(feat, min_train_seasons=3)
    check(not oos.empty, "walk-forward produced out-of-sample predictions",
          f"{len(oos):,} games")

    metrics = evaluate(oos)
    print()
    print(summarize(metrics))
    print()

    naive_mae = float(np.mean(np.abs(oos["margin"])))
    check(metrics["margin_mae"] < naive_mae * 0.85,
          "model beats predicting zero margin",
          f"{metrics['margin_mae']:.2f} vs {naive_mae:.2f}")
    check(0.60 < metrics["win_accuracy"] < 0.95,
          "win-pick accuracy is plausible",
          f"{metrics['win_accuracy']*100:.1f}%")
    check(metrics["win_brier"] < 0.25, "win probabilities beat a coin flip",
          f"Brier {metrics['win_brier']:.4f}")
    check(metrics["margin_mae"] < 15.5, "margin MAE in a realistic band",
          f"{metrics['margin_mae']:.2f}")

    # Calibration. ECE is the sample-weighted headline; the per-bucket check
    # only looks at buckets big enough for the observed rate to mean anything
    # (a 50-game bucket has a ~7pt standard error all by itself).
    check(metrics["win_ece"] < 0.05, "win probabilities are calibrated (ECE)",
          f"ECE {metrics['win_ece']:.4f}")

    cal = pd.DataFrame(metrics["calibration_table"])
    big = cal[cal["n"] >= 100]
    worst = float((big["predicted"] - big["observed"]).abs().max()) if len(big) else 1.0
    check(worst < 0.09, "no large calibration gap in well-populated buckets",
          f"worst {worst:.3f} over {len(big)} buckets")

    # The simulated market is deliberately sharp, so the model should trail it.
    if "mae_vs_market" in metrics:
        check(metrics["market_margin_mae"] < metrics["model_margin_mae_on_lined"],
              "sanity: the sharp simulated market beats the model",
              f"{metrics['market_margin_mae']:.2f} vs "
              f"{metrics['model_margin_mae_on_lined']:.2f}")

    # ---------------------------------------------------------------- artefacts
    section("6. Prediction payload and dashboard")
    model = CFBModel().fit(X, y)
    sample = feat[feat["season"] == 2023].head(9).copy()
    preds = model.predict(sample)
    out = sample.join(preds)

    check(out["home_win_prob"].between(0, 1).all(), "probabilities within [0,1]")
    check(np.allclose(out["pred_home_points"] + out["pred_away_points"],
                      out["pred_total"]), "score split reconciles with total")
    check(np.allclose(out["pred_home_points"] - out["pred_away_points"],
                      out["pred_margin"]), "score split reconciles with margin")

    records = []
    for _, r in out.iterrows():
        margin = float(r["pred_margin"])
        market_margin = -float(r["spread"])
        edge = margin - market_margin
        records.append({
            "game_id": int(r["game_id"]), "week": int(r["week"]),
            "kickoff_utc": r["kickoff"].isoformat(),
            "kickoff_et": r["kickoff"].strftime("%a %I:%M %p ET"),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "neutral_site": bool(r["neutral_site"]),
            "pred_margin": round(margin, 1),
            "pred_total": round(float(r["pred_total"]), 1),
            "pred_home_points": round(float(r["pred_home_points"]), 1),
            "pred_away_points": round(float(r["pred_away_points"]), 1),
            "home_win_prob": round(float(r["home_win_prob"]), 4),
            "market_spread": round(float(r["spread"]), 1),
            "market_total": round(float(r["over_under"]), 1),
            "spread_edge": round(edge, 1),
            "total_edge": round(float(r["pred_total"]) - float(r["over_under"]), 1),
            "spread_tier": "Strong" if abs(edge) >= 6 else ("Lean" if abs(edge) >= 3.5 else None),
            "total_tier": None,
            "spread_play": f"{r['home_team']} {r['spread']:+.1f}" if edge > 0
                           else f"{r['away_team']} {-float(r['spread']):+.1f}",
            "total_play": None,
            "weather": {"temp_f": r["temp_f"], "wind_mph": r["wind_mph"],
                        "precip_in": r["precip_in"], "is_dome": int(r["is_dome"])},
            "weather_text": f"{round(r['temp_f'])}°F, {round(r['wind_mph'])} mph wind",
            "home_rating": round(float(r["home_rating"]), 1),
            "away_rating": round(float(r["away_rating"]), 1),
            "games_played": int(min(r["home_played"], r["away_played"])),
        })

    payload = {
        "generated_at": "2026-09-03T08:00:00-04:00",
        "season": 2023,
        "dates": ["2026-09-05"],
        "games": records,
        "model_metrics": metrics,
    }

    html = render(payload)
    check(len(html) > 6000, "dashboard rendered", f"{len(html):,} bytes")
    check(html.count("<article") == len(records), "every game rendered as a row")
    check("data-theme=\"dark\"" in html and "prefers-color-scheme" in html,
          "dashboard defines both themes")
    check(html.count("<html") == 1 and html.rstrip().endswith("</html>"),
          "dashboard is a complete document")

    out_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "sample.html").write_text(html)
    (out_dir / "sample.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  wrote {out_dir / 'sample.html'}")

    # ---------------------------------------------------------------- verdict
    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 58}")
    print(f"{passed}/{total} checks passed")
    if passed < total:
        print("\nFailures:")
        for ok, label in results:
            if not ok:
                print(f"  - {label}")
    print("=" * 58)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
