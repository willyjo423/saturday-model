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

    # -------------------------------------------------- efficiency (leak-free)
    section("3b. Play-level efficiency")
    from efficiency import EfficiencyEngine, METRICS, normalise_game_stats

    adv = normalise_game_stats(pd.json_normalize(world["advanced"]))
    check(not adv.empty, "advanced game stats normalised",
          f"{len(adv):,} team-games")
    worst_metric = max(adv[f"off_{m}"].isna().mean() for m in METRICS)
    check(worst_metric < 0.02, "every metric resolved from the API shape",
          f"worst {worst_metric:.1%} missing")

    eff = EfficiencyEngine(adv)
    eff_full = eff.stats_before(2022, 8)
    eff_trunc = EfficiencyEngine(
        adv[~((adv["season"] == 2022) & (adv["week"] >= 8))]).stats_before(2022, 8)
    merged_eff = eff_full.merge(eff_trunc, on="team", suffixes=("_f", "_t"))
    eff_diff = float((merged_eff["off_ppa_f"] - merged_eff["off_ppa_t"]).abs().max())
    check(eff_diff < 1e-9, "efficiency identical with future weeks deleted",
          f"max diff {eff_diff:.2e}")

    eff_mid = eff.stats_before(2022, 12).set_index("team")
    common_eff = eff_mid.index.intersection(truth.index)
    eff_corr = float(np.corrcoef(eff_mid.loc[common_eff, "off_ppa"],
                                 truth[common_eff])[0, 1])
    check(eff_corr > 0.75, "efficiency recovers hidden team strength",
          f"r = {eff_corr:.3f}")

    # ---------------------------------------------------------------- features
    section("4. Feature construction")
    weather = FakeWeather(world["games"], world["venues"])
    completed = games[games["completed"]].copy()
    from features import SeasonContext
    ctx = SeasonContext()
    for yr, (talent_df, returning_df) in world["preseason"].items():
        ctx.add_season(yr, talent_df, returning_df)

    feat = build_features(completed, engine, weather, with_weather=True,
                          historical=True, efficiency=eff, context=ctx)

    check(len(feat) == len(completed), "one feature row per game")
    missing_cols = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    check(not missing_cols, "all declared features present", str(missing_cols))
    # Core features must always be present. Efficiency is legitimately absent
    # in week 1 (nothing has been played yet), so it is checked separately.
    from comps import COMP_FEATURES
    from features import CONTEXT_COLUMNS, EFF_EDGE_COLUMNS, EFF_PACE_COLUMNS, EFF_RAW_COLUMNS
    # Efficiency is absent in week 1 and comps are absent in the first season -
    # both legitimately, and both checked on their own terms below.
    derived = set(EFF_EDGE_COLUMNS + EFF_RAW_COLUMNS + EFF_PACE_COLUMNS
                  + COMP_FEATURES)
    core = [c for c in FEATURE_COLUMNS if c not in derived]
    nan_share = feat[core].isna().mean().max()
    worst = feat[core].isna().mean().idxmax()
    check(nan_share < 0.02, "core features essentially free of NaNs",
          f"worst column {worst} at {nan_share:.3%}")

    mid = feat[feat["week"] >= 4]
    eff_present = mid["home_off_ppa"].notna().mean()
    check(eff_present > 0.95, "efficiency present once the season is under way",
          f"{eff_present:.1%} of week-4+ games")
    wk1 = feat[feat["week"] == 1]
    check(len(wk1) == 0 or wk1["home_off_ppa"].isna().all(),
          "efficiency correctly absent in week 1")

    ctx_present = feat[CONTEXT_COLUMNS].notna().mean().min()
    check(ctx_present > 0.95, "talent and returning production attached",
          f"{ctx_present:.1%}")

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

    # ------------------------------------------------------------ comparables
    section("4b. Historical comparables")
    from comps import CompsEngine, attach_comps, learn_axis_weights

    X_pre, y_pre = training_matrix(feat)
    axis_weights = learn_axis_weights(X_pre, y_pre["margin"])
    check(bool(axis_weights), "axis weights learned from a quick booster",
          f"top: {sorted(axis_weights.items(), key=lambda kv: -kv[1])[:3]}"
          if axis_weights else "fell back to defaults")

    comps_engine = CompsEngine(feat, weights=axis_weights)
    check(comps_engine.available, "comps pool built",
          f"{len(comps_engine.history):,} games, {len(comps_engine.axes)} axes")

    feat = attach_comps(feat, comps_engine)
    found = (feat["comp_n"] >= 20).mean()
    check(found > 0.75, "most games find comparables",
          f"{found:.1%} (the earliest season has nothing to look back on)")

    # The critical property: a game must never be its own comparable, and must
    # never draw on anything that happened after it.
    sample_rows = feat[feat["season"] >= 2021].head(60)
    idx, _ = comps_engine.neighbours(sample_rows, k=50)
    pool_time = comps_engine.pool_time
    target_time = sample_rows["season"].to_numpy() * 100 + sample_rows["week"].to_numpy()
    violations = 0
    for i in range(len(sample_rows)):
        picked = idx[i][idx[i] >= 0]
        if len(picked) and (pool_time[picked] >= target_time[i]).any():
            violations += 1
    check(violations == 0, "comparables only ever come from earlier games",
          f"{violations} violations across {len(sample_rows)} games")

    ex = comps_engine.examples(feat.iloc[-1], k=3)
    check(len(ex) > 0 and all("home_team" in e for e in ex),
          "named precedents available for the dashboard",
          f"{len(ex)} returned")

    # Comparables should agree with reality more often than a coin flip.
    graded = feat.dropna(subset=["comp_home_win_rate", "home_win"])
    if len(graded) > 500:
        agree = ((graded["comp_home_win_rate"] > 0.5).astype(int)
                 == graded["home_win"]).mean()
        check(agree > 0.60, "comps alone pick winners better than chance",
              f"{agree:.1%} on {len(graded):,} games")

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

    # Does the new feature group actually pay for itself? Same pipeline, same
    # walk-forward, efficiency and preseason context removed. A feature set is
    # worth keeping only if this number moves the right way.
    section("5b. Are the new features earning their place?")
    plain = build_features(completed, engine, weather, with_weather=True,
                           historical=True, efficiency=None, context=None)
    oos_plain = walk_forward(plain, min_train_seasons=3)
    m_plain = evaluate(oos_plain)
    gain = m_plain["margin_mae"] - metrics["margin_mae"]
    total_gain = m_plain["total_mae"] - metrics["total_mae"]
    print(f"  baseline (ratings + situation + weather only)")
    print(f"  margin MAE  {m_plain['margin_mae']:.3f}  ->  "
          f"{metrics['margin_mae']:.3f}   gain {gain:+.3f} pts")
    print(f"  total  MAE  {m_plain['total_mae']:.3f}  ->  "
          f"{metrics['total_mae']:.3f}   gain {total_gain:+.3f} pts")
    print(f"  log loss    {m_plain['win_logloss']:.4f}  ->  "
          f"{metrics['win_logloss']:.4f}")
    check(gain > 0, "efficiency, context and comps improve margin accuracy",
          f"{gain:+.3f} pts of MAE")
    check(total_gain > 0, "tempo improves total accuracy",
          f"{total_gain:+.3f} pts of MAE")

    # Isolate the comparables specifically: everything else held constant.
    no_comps = feat.copy()
    from comps import COMP_FEATURES as _CF
    for col in _CF:
        no_comps[col] = np.nan
    oos_nc = walk_forward(no_comps, min_train_seasons=3)
    m_nc = evaluate(oos_nc)
    comp_gain = m_nc["margin_mae"] - metrics["margin_mae"]
    print(f"\n  comparables alone: margin MAE {m_nc['margin_mae']:.3f}  ->  "
          f"{metrics['margin_mae']:.3f}   gain {comp_gain:+.3f} pts")
    print(f"  comparables alone: log loss   {m_nc['win_logloss']:.4f}  ->  "
          f"{metrics['win_logloss']:.4f}")
    # Reported, not asserted. Comps earn their place on the dashboard through
    # the distributions and precedents they show; any accuracy gain on top is
    # a bonus, and pretending otherwise would be the kind of number-fitting
    # this backtest exists to prevent.
    metrics["comp_margin_gain"] = round(float(comp_gain), 4)

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

    # ------------------------------------------------------- edge calibration
    section("5c. Edge reliability")
    from edges import BREAKEVEN, EdgeCalibration, confidence_flags

    cal = EdgeCalibration.fit(oos)
    check(cal.fitted, "edge calibration fitted on out-of-sample games",
          f"{cal.n_fitted:,} games, {len(cal.buckets)} buckets")
    print()
    print("  " + cal.summary().replace("\n", "\n  "))
    print()

    # A claimed edge must never be amplified or flipped by the shrinkage.
    probe = np.array([-25.0, -12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0, 25.0])
    shrunk = cal.shrink(probe)
    check(np.all(np.abs(shrunk) <= np.abs(probe) + 1e-9),
          "shrinkage never amplifies an edge")
    check(np.all(np.sign(shrunk) * np.sign(probe) >= 0),
          "shrinkage never flips the side of a play")

    # Does shrinking actually improve the prediction against the line?
    lined = oos.dropna(subset=["spread", "margin", "pred_margin"]).copy()
    lined["market"] = -lined["spread"]
    lined["edge"] = lined["pred_margin"] - lined["market"]
    lined["shrunk_pred"] = lined["market"] + cal.shrink(lined["edge"].to_numpy())
    raw_mae = float(np.mean(np.abs(lined["margin"] - lined["pred_margin"])))
    shr_mae = float(np.mean(np.abs(lined["margin"] - lined["shrunk_pred"])))
    print(f"  margin MAE  raw {raw_mae:.3f}  ->  shrunk {shr_mae:.3f}   "
          f"gain {raw_mae - shr_mae:+.3f} pts")
    check(shr_mae <= raw_mae + 0.02,
          "shrinking edges toward the market does not hurt accuracy",
          f"{raw_mae - shr_mae:+.3f} pts")

    # The observation that started this: are big edges worse than small ones?
    if len(cal.buckets) >= 3:
        small = cal.buckets[0]
        large = cal.buckets[-1]
        print(f"  smallest bucket ({small['lo']:.1f}-{small['hi']:.1f}) "
              f"ATS {small['ats']*100:.1f}% on {small['n']:,}")
        print(f"  largest  bucket ({large['lo']:.1f}-{large['hi']:.1f}) "
              f"ATS {large['ats']*100:.1f}% on {large['n']:,}")
        keeps_less = large["realized_fraction"] < small["realized_fraction"]
        print(f"  large edges keep {'less' if keeps_less else 'more'} of "
              f"themselves than small ones")

    flags = confidence_flags(feat.iloc[0])
    check(isinstance(flags, list), "confidence flags computed without error",
          f"{len(flags)} on a sample game")

    round_trip = EdgeCalibration.from_json(cal.to_json())
    check(abs(round_trip.a - cal.a) < 1e-9 and round_trip.fitted,
          "calibration survives a save/load round trip")

    # -------------------------------------------------- comparables reliability
    section("5d. Do the comparables actually predict covering?")
    from edges import CompsCalibration

    ccal = CompsCalibration.fit(oos)
    check(ccal.fitted, "comparables calibration fitted",
          f"{ccal.n_fitted:,} graded games")
    print()
    print("  " + ccal.summary().replace("\n", "\n  "))
    print()

    # The calibrated rate must never be more confident than the raw one.
    probes = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    cals = [ccal.calibrated_rate(p) for p in probes]
    print("  raw -> calibrated: " +
          "  ".join(f"{p:.0%}->{c:.0%}" for p, c in zip(probes, cals)))
    check(all(abs(c - 0.5) <= abs(p - 0.5) + 1e-6
              for p, c in zip(probes, cals)),
          "calibration only ever pulls a rate toward a coin flip")
    check(all(a <= b + 1e-9 for a, b in zip(cals, cals[1:])),
          "calibrated rate rises with the raw rate")

    # Does the raw rate carry signal at all? Compare the top and bottom bands.
    if len(ccal.buckets) >= 2:
        low, high = ccal.buckets[0], ccal.buckets[-1]
        lift = high["realized"] - low["realized"]
        print(f"  lowest band said {low['predicted']:.0%}, actually covered "
              f"{low['realized']:.1%} (n={low['n']:,})")
        print(f"  highest band said {high['predicted']:.0%}, actually covered "
              f"{high['realized']:.1%} (n={high['n']:,})")
        print(f"  lift from bottom band to top: {lift * 100:+.1f} points")
        # Reported, not asserted. If the comparables carry no signal on real
        # data, the tiering will simply stop offering plays - which is the
        # correct behaviour, and better than a test that pretends otherwise.
        metrics["comps_lift"] = round(float(lift), 4)

    assess = ccal.assess(0.65)
    check(assess["side"] in ("home", "away") and assess["confidence"] is not None,
          "assess returns a usable verdict",
          f"side={assess['side']} conf={assess['confidence']:.0%} "
          f"tier={assess['tier']}")
    check(CompsCalibration.from_json(ccal.to_json()).fitted,
          "comparables calibration survives a round trip")

    # ---------------------------------------------------------------- artefacts
    section("6. Prediction payload and dashboard")
    model = CFBModel().fit(X, y)
    sample = feat[feat["season"] == 2023].head(9).copy()
    preds = model.predict(sample)
    out = sample.join(preds)

    check(out["home_win_prob"].between(0, 1).all(), "probabilities within [0,1]")

    # The bug that started this: a huge favourite was coming back at 58%.
    # Win probability must rise monotonically with margin and reach near
    # certainty for a blowout, including margins larger than any in training.
    ladder = [(0, 0.47, 0.53), (7, 0.62, 0.75), (14, 0.75, 0.88),
              (21, 0.85, 0.95), (35, 0.94, 0.995), (50, 0.96, 0.999)]
    probs = [model._win_prob(np.array([float(m)]))[0] for m, _, _ in ladder]
    in_band = all(lo <= p <= hi for p, (_, lo, hi) in zip(probs, ladder))
    check(in_band, "win probability sane across the margin ladder",
          " ".join(f"{m}pt={p*100:.0f}%" for (m, _, _), p in zip(ladder, probs)))
    check(all(a < b for a, b in zip(probs, probs[1:])),
          "win probability rises monotonically with margin")

    # Uncertainty should depend on how much the ratings had to go on. A week-2
    # forecast built on preseason priors deserves a wider spread - and a less
    # confident probability - than a week-12 one.
    curve = model.sigma_by_played
    check(bool(curve), "residual spread fitted by games played",
          " ".join(f"{c['lo']:.0f}-{c['hi']:.0f}:{c['sigma']:.1f}" for c in curve))
    if curve:
        thin, thick = curve[0]["sigma"], curve[-1]["sigma"]
        print(f"  thin evidence sigma {thin:.2f} vs thick {thick:.2f} "
              f"({thin - thick:+.2f})")
        sigmas = [c["sigma"] for c in curve]
        check(all(a >= b - 1e-9 for a, b in zip(sigmas, sigmas[1:])),
              "uncertainty never rises as evidence accumulates",
              " -> ".join(f"{v:.1f}" for v in sigmas))

        # The same margin must yield a less confident probability when the
        # ratings behind it are thin.
        p_thin = model._win_prob(np.array([10.0]), np.array([float(curve[0]["lo"])]))[0]
        p_thick = model._win_prob(np.array([10.0]), np.array([float(curve[-1]["lo"])]))[0]
        print(f"  a 10-pt edge reads {p_thin*100:.1f}% on thin evidence, "
              f"{p_thick*100:.1f}% on thick")
        check(p_thin <= p_thick + 1e-9,
              "thin evidence never produces more confidence than thick",
              f"{p_thin*100:.1f}% vs {p_thick*100:.1f}%")

    sig = model.predict(X)["margin_sigma"]
    check(sig.notna().all() and (sig > 0).all(),
          "every prediction carries its own uncertainty",
          f"{sig.min():.1f}-{sig.max():.1f} pts")

    # Predictions must be able to exceed the training range, which tree
    # averaging alone can never do - that is what the linear baseline buys.
    all_preds = model.predict(X)["pred_margin"]
    reach = max(abs(all_preds.max()), abs(all_preds.min()))
    check(reach > 25, "model can express a lopsided game",
          f"largest predicted margin {reach:.1f} pts, "
          f"sd {all_preds.std():.1f} vs outcome sd {y['margin'].std():.1f}")

    # A model saved before a feature-set change must say so plainly rather
    # than dying inside pandas with a bare KeyError.
    from model import FeatureMismatchError
    stale = CFBModel(features=list(model.features) + ["a_feature_we_dropped"])
    stale.margin, stale.total = model.margin, model.total
    stale.margin_line, stale.total_line = model.margin_line, model.total_line
    stale.wp_coef, stale.wp_intercept = model.wp_coef, model.wp_intercept
    stale.trained_seasons = list(model.trained_seasons)
    try:
        stale.predict(sample)
        mismatch_ok = False
        detail = "no error raised"
    except FeatureMismatchError as exc:
        mismatch_ok = "Bootstrap" in str(exc) and "a_feature_we_dropped" in str(exc)
        detail = "names the column and the fix"
    except Exception as exc:  # noqa: BLE001
        mismatch_ok = False
        detail = f"wrong exception: {type(exc).__name__}"
    check(mismatch_ok, "stale model gives an actionable error", detail)
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
