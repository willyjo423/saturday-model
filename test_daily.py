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

    # Enrichment endpoints the priors try and can live without.
    def sp_ratings(self, year):
        raise _Unavailable("stub: no SP+")

    def talent(self, year):
        raise _Unavailable("stub: no talent")

    def returning_production(self, year):
        raise _Unavailable("stub: no returning production")


from api import CFBDError as _Unavailable  # noqa: E402


class StubWeather:
    """Forecast-shaped weather that varies by venue, no network."""

    def at_kickoff(self, lat, lon, kickoff, is_dome=False, historical=None):
        if is_dome:
            return {"temp_f": 70.0, "humidity": 50.0, "precip_in": 0.0,
                    "wind_mph": 0.0, "gust_mph": 0.0, "is_dome": 1}
        seed = 0.0 if lat is None or pd.isna(lat) else float(lat)
        return {"temp_f": 40 + (seed % 40), "humidity": 55.0,
                "precip_in": 0.05 if int(seed) % 7 == 0 else 0.0,
                "wind_mph": 3 + (seed % 17), "gust_mph": 8 + (seed % 20),
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
    from dataset import add_rest_and_travel
    games = add_rest_and_travel(games)

    priors = PreseasonPriors()
    seed = RatingsEngine(games, PreseasonPriors())
    for year in sorted(games["season"].unique()):
        prior_final = seed.final_ratings(year - 1)
        priors.build(int(year), None, None, None, prior_final)
    engine = RatingsEngine(games, priors)

    feat = build_features(games[games["completed"]], engine,
                          FakeWeather(world["games"], world["venues"]),
                          with_weather=True, historical=True)
    X, y = training_matrix(feat)
    model = CFBModel().fit(X, y)
    joblib.dump(model, config.MODELS / "cfb_model.joblib")


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

    check(all(0 <= g["home_win_prob"] <= 1 for g in games), "probabilities in range")
    check(bool(games) and all(g["pred_total"] > 10 for g in games),
          "totals are plausible",
          f"min {min((g['pred_total'] for g in games), default=0):.1f}")

    # Edge, tier and play text must agree with each other.
    consistent = True
    for g in games:
        edge = g["spread_edge"]
        if edge is None:
            continue
        expected = None
        for threshold, label in config.EDGE_TIERS:
            if abs(edge) >= threshold:
                expected = label
                break
        if g["spread_tier"] != expected:
            consistent = False
        side = g["home_team"] if edge > 0 else g["away_team"]
        if g["spread_play"] and not g["spread_play"].startswith(side):
            consistent = False
    check(consistent, "edge, tier and recommended side agree")

    ordered = [abs(g["spread_edge"]) for g in games if g["spread_edge"] is not None]
    check(ordered == sorted(ordered, reverse=True),
          "slate sorted by size of disagreement")

    check(all(g["weather_text"] for g in games), "weather attached to every game")

    # Serialisable, which is what the workflow commits.
    blob = json.dumps(payload, default=str)
    check(len(blob) > 500, "payload serialises to JSON", f"{len(blob):,} bytes")

    from dashboard import render
    html = render(payload)
    check(html.count("<article") == len(games), "dashboard renders the whole slate")
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
