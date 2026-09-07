"""Exercise the daily run end to end with the API and weather stubbed out.

This is the script that runs unattended every morning, so it gets its own
test: schedule detection, slate selection, line joining, prediction, edge
tiering, JSON payload, and dashboard render — the whole path, offline.

    python -m tests.test_daily
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd


import config, daily, schema  # noqa: E402
from dataset import clean_games  # noqa: E402
from features import build_features, training_matrix  # noqa: E402
from model import CFBModel  # noqa: E402
from ratings import PreseasonPriors, RatingsEngine  # noqa: E402
from simulate import FakeWeather, build_world  # noqa: E402

results: list[tuple[bool, str]] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    results.append((bool(cond), label))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


class StubClient:
    """Stands in for CFBDClient, serving the simulated world."""

    def __init__(self, world, *args, **kwargs):
        self.world = world
        self.calls_made = 0

    def venues(self):
        self.calls_made += 1
        return pd.json_normalize(self.world["venues"])

    def all_games(self, year, classification="fbs"):
        self.calls_made += 1
        rows = [g for g in self.world["games"] if g["season"] == year]
        return pd.json_normalize(rows) if rows else pd.DataFrame()

    def lines(self, year, week=None, season_type="regular"):
        self.calls_made += 1
        if season_type != "regular":
            return pd.DataFrame()
        rows = [entry for entry in self.world["lines"] if entry["season"] == year]
        return pd.json_normalize(rows) if rows else pd.DataFrame()

    def advanced_game_stats(self, year, week=None, season_type="regular"):
        self.calls_made += 1
        if season_type != "regular":
            return pd.DataFrame()
        rows = [r for r in self.world.get("advanced", [])
                if r["season"] == year]
        return pd.json_normalize(rows) if rows else pd.DataFrame()

    def talent(self, year):
        self.calls_made += 1
        pre = self.world.get("preseason", {}).get(year)
        if pre is None:
            raise _Unavailable("stub: no talent for that season")
        return pre[0]

    def returning_production(self, year):
        self.calls_made += 1
        pre = self.world.get("preseason", {}).get(year)
        if pre is None:
            raise _Unavailable("stub: no returning production")
        return pre[1]

    # SP+ genuinely absent, so the priors path that copes without it is tested.
    def sp_ratings(self, year):
        raise _Unavailable("stub: no SP+")


from api import CFBDError as _Unavailable  # noqa: E402


class StubWeather:
    """Forecast-shaped weather that varies by venue, no network."""

    def __init__(self, budget_seconds: float = 120.0):
        self.budget_seconds = budget_seconds

    def prefetch(self, games, historical=True):
        """Matches the real service's interface; nothing to warm."""

    def coverage(self) -> str:
        return "stub weather"

    def at_kickoff(self, lat, lon, kickoff, is_dome=False, historical=None):
        if is_dome:
            return {"temp_f": 70.0, "humidity": 50.0, "precip_in": 0.0,
                    "wind_mph": 0.0, "is_dome": 1}
        seed = 0.0 if lat is None or pd.isna(lat) else float(lat)
        return {"temp_f": 40 + (seed % 40), "humidity": 55.0,
                "precip_in": 0.05 if int(seed) % 7 == 0 else 0.0,
                "wind_mph": 3 + (seed % 17),
                "is_dome": 0}


def train_stub_model(world) -> None:
    """Train a small model on the simulated past so daily.run can load it."""
    v = schema.normalise(pd.json_normalize(world["venues"]), schema.VENUE_FIELDS)
    for c in ("lat", "lon", "elevation", "capacity"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["dome"] = v["dome"].fillna(False).astype(bool)
    v["venue_id"] = pd.to_numeric(v["venue_id"], errors="coerce")

    g = schema.normalise(pd.json_normalize(world["games"]), schema.GAME_FIELDS)
    games = clean_games(g, v)
    from dataset import _best_line, add_rest_and_travel
    games = add_rest_and_travel(games)

    # Join the simulated market, so the stub can fit an edge calibration the
    # same way the real training run does.
    line_rows = [{"game_id": e["id"], **_best_line(e["lines"])}
                 for e in world["lines"]]
    games = games.merge(pd.DataFrame(line_rows), on="game_id", how="left")

    priors = PreseasonPriors()
    seed = RatingsEngine(games, PreseasonPriors())
    for year in sorted(games["season"].unique()):
        prior_final = seed.final_ratings(year - 1)
        priors.build(int(year), None, None, None, prior_final)
    engine = RatingsEngine(games, priors)

    from efficiency import EfficiencyEngine, normalise_game_stats
    from features import SeasonContext
    eff = EfficiencyEngine(normalise_game_stats(
        pd.json_normalize(world.get("advanced", []))))
    ctx = SeasonContext()
    for yr, (talent_df, returning_df) in world.get("preseason", {}).items():
        ctx.add_season(yr, talent_df, returning_df)

    feat = build_features(games[games["completed"]], engine,
                          FakeWeather(world["games"], world["venues"]),
                          with_weather=True, historical=True,
                          efficiency=eff, context=ctx)

    # Attach comparables and persist the table, exactly as the bootstrap job
    # does - the daily run reads this file to find precedents for today's games.
    from comps import CompsEngine, attach_comps
    feat = attach_comps(feat, CompsEngine(feat))
    from storage import save_table
    save_table(feat, config.DATA / "training")

    X, y = training_matrix(feat)
    model = CFBModel().fit(X, y)
    joblib.dump(model, config.MODELS / "cfb_model.joblib")

    # Write an edge calibration so the daily run exercises the shrink-and-tier
    # path. Fitted in-sample here purely to produce a well-formed artefact -
    # the real one comes from train.py's walk-forward backtest.
    from edges import EdgeCalibration
    graded = y.join(model.predict(X)[["pred_margin"]])
    cal = EdgeCalibration.fit(graded)
    (config.MODELS / "edge_calibration.json").write_text(cal.to_json())


def main() -> int:
    # Seasons 2024-2025 are history the model learns from; 2026 is live.
    world = build_world([2024, 2025, 2026])

    # Pick the target from the simulated schedule itself: the busiest Eastern
    # calendar day in week 3 of 2026. That mirrors how the real run works -
    # the date leads, and whatever is on it is the slate.
    et_dates = pd.Series([
        pd.Timestamp(g["startDate"]).tz_convert(daily.ET).date()
        for g in world["games"] if g["season"] == 2026 and g["week"] == 3
    ])
    target = et_dates.mode().iloc[0]
    print(f"Simulating a live run for {target}\n" + "-" * 46)

    # Anything kicking off on or after the target morning is still unplayed.
    cutoff = pd.Timestamp(target, tz=daily.ET)
    for g in world["games"]:
        if pd.Timestamp(g["startDate"]) >= cutoff:
            g["homePoints"] = None
            g["awayPoints"] = None
            g["completed"] = False

    history = {**world, "games": [g for g in world["games"] if g["season"] < 2026]}
    train_stub_model(history)
    check((config.MODELS / "cfb_model.joblib").exists(), "stub model trained and saved")

    # Patch the network out.
    daily.CFBDClient = lambda *a, **k: StubClient(world)
    daily.WeatherService = StubWeather
    import build as build_mod
    build_mod.CFBDClient = daily.CFBDClient

    check(daily.current_season(date(2026, 9, 5)) == 2026, "season detected for September")
    check(daily.current_season(date(2027, 1, 8)) == 2026, "January bowls map to prior season")

    payload = daily.run([target])

    games = payload["games"]
    check(len(games) > 0, "slate found without any manual input", f"{len(games)} games")
    check(payload["season"] == 2026, "correct season on the payload")

    et_days = {pd.Timestamp(g["kickoff_utc"]).tz_convert(daily.ET).date()
               for g in games if g["kickoff_utc"]}
    check(et_days == {target}, "every game is on the requested Eastern date",
          str(sorted(str(d) for d in et_days)))

    with_line = [g for g in games if g["market_spread"] is not None]
    check(len(with_line) == len(games), "market lines joined to the whole slate",
          f"{len(with_line)}/{len(games)}")

    check(all(0 <= g["model"]["home_win_prob"] <= 1 for g in games),
          "model probabilities still in range (kept in the payload)")
    check(bool(games) and all(g["model"]["total"] > 10 for g in games),
          "model totals are plausible",
          f"min {min((g['model']['total'] for g in games), default=0):.1f}")

    # --- comparables are now the headline -------------------------------
    with_comps = [g for g in games if g.get("comps")]
    check(len(with_comps) > len(games) * 0.8,
          "comparables found for today's slate",
          f"{len(with_comps)}/{len(games)} games")

    c = with_comps[0]["comps"]
    check(c["margin_p25"] <= c["margin_median"] <= c["margin_p75"],
          "comp distribution is ordered",
          f"p25 {c['margin_p25']} med {c['margin_median']} p75 {c['margin_p75']}")
    check(c["cover_rate"] is not None and 0.0 <= c["cover_rate"] <= 1.0,
          "cover rate present and in range", f"{c['cover_rate']:.0%}")
    check(c["n"] >= 20, "sample size reported", f"{c['n']} comps")

    # Every offered play must name a real side, at that side's real price,
    # and must carry a tier.
    consistent = True
    for g in games:
        play = g.get("play")
        if not play:
            continue
        if play["team"] not in (g["home_team"], g["away_team"]):
            consistent = False
        if not play.get("tier"):
            consistent = False
        # The side must match which way the comparables leaned.
        side_team = (g["home_team"] if g["comps"]["side"] == "home"
                     else g["away_team"])
        if play["team"] != side_team:
            consistent = False
        # And the quoted number must be that team's own line.
        expected = (f'{g["market_spread"]:+.1f}' if play["team"] == g["home_team"]
                    else f'{-g["market_spread"]:+.1f}')
        if play["line"] != expected:
            consistent = False
    check(consistent, "each play names the right side at the right price")

    # A calibrated rate must never be more confident than the raw one.
    tamed = all(
        abs(g["comps"]["calibrated_rate"] - 0.5) <= abs(g["comps"]["cover_rate"] - 0.5) + 1e-6
        for g in with_comps
        if g["comps"].get("calibrated_rate") is not None
        and g["comps"].get("cover_rate") is not None)
    check(tamed, "calibration never increases confidence beyond the raw rate")

    keys = [({"Strong": 0, "Lean": 1, "Slight": 2}.get(
                (g.get("play") or {}).get("tier"), 3),
             -((g.get("comps") or {}).get("confidence") or 0.0))
            for g in games]
    check(keys == sorted(keys), "slate ordered by tier, then by confidence")

    check(all(g["weather_text"] for g in games), "weather attached to every game")

    ex = with_comps[0].get("comp_examples") or []
    check(len(ex) == 5, "five precedents per game", f"{len(ex)} returned")
    check(all(e["season"] < 2026 for e in ex),
          "precedents are all from earlier seasons")
    check(all({"home_points", "away_points", "home_spread", "home_covered"}
              <= set(e) for e in ex),
          "precedents carry score, price and result")

    # Serialisable, which is what the workflow commits.
    blob = json.dumps(payload, default=str)
    check(len(blob) > 500, "payload serialises to JSON", f"{len(blob):,} bytes")

    from dashboard import render
    html = render(payload)
    check(html.count("<article") == len(games), "dashboard renders the whole slate")

    # The precedent table is unreadable without saying which side each
    # historical team stands in for, and whose line is being shown.
    check("stands in for" in html and "The line shown is the home team" in html,
          "precedents explain the role mapping")
    check(">Season<" in html and ">Final<" in html and ">Covered<" in html,
          "precedent table has column headers")
    check("Home line (here" in html,
          "precedent line column names this game's own number")

    # Colour is only meaningful when there is a side to support.
    no_play = [g for g in games if not g.get("play")]
    if no_play:
        from dashboard import _precedents
        frag = _precedents(no_play[0])
        check('class="rs yes"' not in frag and 'class="rs no"' not in frag,
              "no-play cards leave precedent results uncoloured")
        check("this pick needs" not in frag,
              "no-play cards drop the pick-relative legend")
    out = config.DOCS / "sample_daily.html"
    out.write_text(html)
    print(f"\n  wrote {out}")

    # An empty day should be handled, not crash.
    quiet = daily.run([date(2026, 7, 14)])
    check(quiet["games"] == [], "an empty slate returns cleanly")
    empty_html = render(quiet)
    check("No games scheduled" in empty_html, "empty slate renders a sensible page")

    passed = sum(1 for ok, _ in results if ok)
    print(f"\n{'=' * 46}\n{passed}/{len(results)} checks passed")
    if passed < len(results):
        for ok, label in results:
            if not ok:
                print(f"  - {label}")
    print("=" * 46)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
