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

    check(all(0 <= g["forecast"]["home_win_prob"] <= 1 for g in games),
          "forecast probabilities in range")
    check(bool(games) and all(g["forecast"]["total"] > 10 for g in games),
          "forecast totals are plausible",
          f"min {min((g['forecast']['total'] for g in games), default=0):.1f}")
    check(all(g["forecast"]["home_points"] + g["forecast"]["away_points"]
              == round(g["forecast"]["total"]) or
              abs(g["forecast"]["home_points"] + g["forecast"]["away_points"]
                  - g["forecast"]["total"]) <= 1.0
              for g in games),
          "projected score reconciles with the projected total")

    # No recommendation should survive anywhere in the payload.
    check(not any("play" in g for g in games),
          "payload carries no play recommendations")

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



    kicks = [g["kickoff_utc"] or "" for g in games]
    check(kicks == sorted(kicks), "slate reads in kickoff order")

    check(all(g["weather_text"] for g in games), "weather attached to every game")

    # All three market summaries must be present and mutually consistent.
    check(c.get("over_rate") is not None and 0.0 <= c["over_rate"] <= 1.0,
          "over/under rate present", f"{c['over_rate']:.0%} went over")
    check(c.get("home_win_rate") is not None and 0.0 <= c["home_win_rate"] <= 1.0,
          "moneyline win rate present", f"home won {c['home_win_rate']:.0%}")
    check(c.get("cover_n") and c["cover_n"] <= c["n"],
          "graded cover count excludes pushes",
          f"{c['cover_n']} graded of {c['n']} comps")

    check(c.get("home_cover_by") is not None and c.get("away_cover_by") is not None,
          "average cover margin computed for both sides",
          f"home {c['home_cover_by']:.1f} / away {c['away_cover_by']:.1f}")
    check(c.get("home_blowout") is not None and c.get("away_blowout") is not None,
          "blowout rates computed for both sides",
          f"{c['home_blowout']:.0%} / {c['away_blowout']:.0%}")

    # No signed margin may appear as a bare number in the outcome band - the
    # minus sign means "lost by" there and "favoured by" on the market row.
    import re as _re
    from dashboard import _band

    def _plain(g):
        return " ".join(_re.sub("<[^>]+>", "", _band(g)).split())

    # The sign convention was genuinely ambiguous: on most cards the home team
    # is also the favourite, so "negative" reads as both "home lost" and
    # "underdog" and the two never disagree - until the game where they do.
    # All three orientations must now be unmistakable without a sign.
    fav_home = _plain({"home_team": "Alabama", "away_team": "East Carolina",
                       "comps": {"margin_p25": 14.0, "margin_p75": 35.0,
                                 "margin_median": 24.0}})
    check("Alabama winning by 14 to 35" in fav_home,
          "home favourite: one team, one range", fav_home[-46:])

    dog_home = _plain({"home_team": "Nevada", "away_team": "Western Kentucky",
                       "comps": {"margin_p25": -14.0, "margin_p75": 7.0,
                                 "margin_median": -3.0}})
    check("Western Kentucky winning by 14" in dog_home
          and "Nevada winning by 7" in dog_home,
          "straddles zero: both teams named", dog_home[-62:])

    fav_away = _plain({"home_team": "Rice", "away_team": "Texas",
                       "comps": {"margin_p25": -31.0, "margin_p75": -9.0,
                                 "margin_median": -20.0}})
    check("Texas winning by 9 to 31" in fav_away,
          "away favourite: named correctly, not inverted", fav_away[-44:])

    check(not _re.search(r"[+\u2212-]\d+ to [+-]?\d+", _plain(with_comps[0])),
          "no signed margin survives anywhere in the band")

    band = _band(with_comps[0])
    check("wins &#8594;" in band and "&#8592;" in band,
          "band carries direction labels at both ends")

    # The total range, stated the same way and sitting under the margin one.
    totals = {"home_team": "Alabama", "away_team": "East Carolina",
              "comps": {"margin_p25": 14.0, "margin_p75": 35.0,
                        "margin_median": 24.0,
                        "total_p25": 48.0, "total_p75": 63.0}}
    plain_totals = _plain(totals)
    check("added up to between 48 and 63" in plain_totals,
          "total range is shown", plain_totals[-52:])
    check(_band(totals).index("band-label total")
          > _band(totals).index("Half finished"),
          "total line sits under the margin line, not beside it")
    tail = plain_totals.split("added up to")[-1]
    check("Alabama" not in tail and "East Carolina" not in tail,
          "no team named on the total line - a total has no direction")

    no_totals = dict(totals)
    no_totals["comps"] = {k: v for k, v in totals["comps"].items()
                          if not k.startswith("total_")}
    check("added up to" not in _plain(no_totals),
          "an absent total range is omitted, not invented")

    from dashboard import _comps_table
    tbl = _comps_table(with_comps[0])
    check("Spread" in tbl and "Total" in tbl and "Moneyline" in tbl,
          "summary shows all three markets")
    check(f"Across all {c['n']} comparable games" in tbl,
          "summary states the full sample size")
    check("Covers came by" in tbl and "Won by 14 or more" in tbl,
          "summary reports magnitude, not just frequency")

    # A tiny gap must not be reported as an advantage.
    from dashboard import _magnitude
    probe = dict(with_comps[0])
    probe["comps"] = {**probe["comps"], "home_cover_by": 11.7,
                      "away_cover_by": 11.5}
    check("neither side" in _magnitude(probe),
          "a sub-point gap is called even, not an edge")
    probe["comps"] = {**probe["comps"], "home_cover_by": 14.0,
                      "away_cover_by": 9.0}
    check("bigger outcomes belong to" in _magnitude(probe),
          "a real gap does name the side")

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
    from dashboard import _precedents
    frag = _precedents(games[0])
    check('class="rs yes"' not in frag and 'class="rs no"' not in frag,
          "precedent results are reported, never scored for a pick")
    check("this pick needs" not in frag, "no pick-relative legend anywhere")
    check("no play" not in html.lower() and "Strong &middot;" not in html,
          "dashboard contains no play/no-play language")
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
