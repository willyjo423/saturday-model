"""The zero-input daily run.

    python daily.py              # today's slate
    python daily.py --date 2026-09-05
    python daily.py --days 3     # today plus the next two days

It works out the season on its own, pulls the full schedule, picks the games
kicking off on the target date(s), fetches live venue weather, and writes both
a JSON payload and a standalone HTML dashboard. Nothing to submit.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import config, dataset
from api import CFBDClient, CFBDError, MissingKeyError
from build import build_priors
from comps import CompsEngine, attach_comps
from edges import CompsCalibration, EdgeCalibration, confidence_flags
from efficiency import EfficiencyEngine, load_team_game_stats, pool_non_fbs
from model import FeatureMismatchError
from features import SeasonContext, build_features
from ratings import RatingsEngine
from storage import load_table
from weather import WeatherService, describe

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MODEL_PATH = config.MODELS / "cfb_model.joblib"


def current_season(today: date) -> int:
    """College football seasons straddle the new year; bowls in January
    belong to the previous season."""
    return today.year if today.month >= 7 else today.year - 1


def _comp_summary(r, assess: dict) -> dict | None:
    """What happened in the most similar historical matchups.

    The card leads with this, so it carries both the raw observed rate and the
    calibrated one - the raw rate is what the neighbours literally did, and the
    calibrated rate is what that has been worth in practice, which is lower
    because 200 overlapping neighbours are nowhere near 200 independent games.
    """
    n = r.get("comp_n")
    if n is None or pd.isna(n) or n < 20:
        return None

    def val(key, digits=1):
        v = r.get(key)
        return None if v is None or pd.isna(v) else round(float(v), digits)

    bucket = assess.get("bucket") or {}
    return {
        "n": int(n),
        "cover_rate": val("comp_home_cover_rate", 3),
        "home_cover_by": val("comp_home_cover_by"),
        "away_cover_by": val("comp_away_cover_by"),
        "home_blowout": val("comp_home_blowout", 3),
        "away_blowout": val("comp_away_blowout", 3),
        "cover_n": (None if r.get("comp_cover_n") is None
                    or pd.isna(r.get("comp_cover_n"))
                    else int(r["comp_cover_n"])),
        "over_n": (None if r.get("comp_over_n") is None
                   or pd.isna(r.get("comp_over_n"))
                   else int(r["comp_over_n"])),
        "calibrated_rate": (None if assess.get("calibrated_rate") is None
                            else round(assess["calibrated_rate"], 3)),
        "side": assess.get("side"),
        "confidence": (None if assess.get("confidence") is None
                       else round(assess["confidence"], 3)),
        "historical_rate": bucket.get("realized"),
        "historical_n": bucket.get("n"),
        "margin_median": val("comp_margin_median"),
        "margin_p25": val("comp_margin_p25"),
        "margin_p75": val("comp_margin_p75"),
        "total_median": val("comp_total_median"),
        "total_p25": val("comp_total_p25"),
        "total_p75": val("comp_total_p75"),
        "home_win_rate": val("comp_home_win_rate", 3),
        "over_rate": val("comp_over_rate", 3),
        "agreement": val("comp_agreement", 3),
        "price_spread": val("comp_spread_spread", 1),
    }


def tier_for(edge: float) -> str | None:
    for threshold, label in config.EDGE_TIERS:
        if abs(edge) >= threshold:
            return label
    return None


def run(target_dates: list[date], api_key: str | None = None,
        with_weather: bool = True, weather_budget: float = 120.0) -> dict:
    client = CFBDClient(api_key=api_key)
    season = current_season(target_dates[0])
    log.info("Season %s; target dates %s", season,
             ", ".join(d.isoformat() for d in target_dates))

    venues = dataset.load_venues(client)

    # Two seasons: the current one for ratings, the previous one so the
    # preseason prior has last year's finish to lean on.
    raw = dataset.load_games(client, [season - 1, season])
    games = dataset.clean_games(raw, venues)
    if games.empty:
        raise SystemExit("No games returned from the API.")
    games = dataset.add_rest_and_travel(games)

    context = SeasonContext()
    priors = build_priors(client, [season], games, context=context)
    engine = RatingsEngine(games, priors)

    # Play-level efficiency for the current season only - last season's games
    # feed the preseason prior, not the in-season efficiency solve.
    fbs_teams = set(games.loc[games["home_is_fbs"], "home_team"]) | \
                set(games.loc[games["away_is_fbs"], "away_team"])
    team_games = pool_non_fbs(load_team_game_stats(client, [season]), fbs_teams)
    efficiency = EfficiencyEngine(team_games)

    # --- select the slate ---
    kick_et = games["kickoff"].dt.tz_convert(ET)
    on_date = kick_et.dt.date.isin(set(target_dates))
    slate = games.loc[on_date & (games["season"] == season)].copy()
    log.info("%d games on the slate", len(slate))
    if slate.empty:
        return {"generated_at": datetime.now(ET).isoformat(),
                "season": season,
                "dates": [d.isoformat() for d in target_dates],
                "games": [], "note": "No games scheduled."}

    # --- betting lines (one call covers the whole season, then cached) ---
    try:
        lines = dataset.load_lines(client, [season])
    except CFBDError as exc:
        log.warning("betting lines unavailable, edges will be blank: %s", exc)
        lines = pd.DataFrame(columns=["game_id", "spread", "over_under", "provider"])
    slate = slate.merge(lines, on="game_id", how="left")
    log.info("%d of %d games have a market line",
             int(slate["spread"].notna().sum()), len(slate))

    # --- features with live forecast weather ---
    # Capped at two minutes. Weather sharpens a prediction; it must never be
    # the reason a morning run fails to produce one.
    weather = WeatherService(budget_seconds=weather_budget) if with_weather else None
    if weather is not None:
        weather.prefetch(slate, historical=False)
        log.info("Weather: %s", weather.coverage())
    feat = build_features(slate, engine, weather,
                          with_weather=with_weather, historical=False,
                          efficiency=efficiency, context=context)

    # --- historical comparables -------------------------------------------
    # The pool is the committed training table, so today's games are matched
    # against a decade of finished ones. Absent it, comps go blank and the
    # model falls back on everything else.
    comps_engine = None
    history = load_table(config.DATA / "training")
    if history is not None:
        try:
            weights_path = config.MODELS / "comp_weights.json"
            weights = (json.loads(weights_path.read_text())
                       if weights_path.exists() else None)
            comps_engine = CompsEngine(history, weights=weights)
            feat = attach_comps(feat, comps_engine)
        except Exception as exc:  # noqa: BLE001 - comps are enrichment
            log.warning("comparables unavailable: %s", exc)
    else:
        log.info("no training table found; skipping comparables")

    # How much of a claimed edge to believe, learned from the backtest.
    cal_path = config.MODELS / "edge_calibration.json"
    calibration = (EdgeCalibration.from_json(cal_path.read_text())
                   if cal_path.exists() else EdgeCalibration())
    if calibration.fitted:
        log.info("edge calibration loaded (fitted on %d games)",
                 calibration.n_fitted)
    else:
        log.info("no edge calibration found; edges used unshrunk")

    comps_cal_path = config.MODELS / "comps_calibration.json"
    comps_cal = (CompsCalibration.from_json(comps_cal_path.read_text())
                 if comps_cal_path.exists() else CompsCalibration())
    if comps_cal.fitted:
        log.info("comparables calibration loaded (fitted on %d games, "
                 "slope %.2f)", comps_cal.n_fitted, comps_cal.slope)
    else:
        log.info("no comparables calibration found; rates shown unadjusted")

    model = joblib.load(MODEL_PATH)
    preds = model.predict(feat)
    out = feat.join(preds)

    # --- assemble ---
    records = []
    for _, r in out.iterrows():
        margin = float(r["pred_margin"])
        total = float(r["pred_total"])
        prob = float(r["home_win_prob"])
        spread = r["spread"]
        ou = r["over_under"]

        market_margin = -float(spread) if pd.notna(spread) else None
        # Kept as context, not as a recommendation. Measured against 5,000+
        # out-of-sample games, neither the model's disagreement with the
        # closing line nor the comparables' cover rate predicted covering -
        # every band came back at 50%. So the difference is reported as
        # information about where this forecast sits relative to the market,
        # and nothing is dressed up as a play.
        raw_edge = round(margin - market_margin, 1) if market_margin is not None else None
        total_edge = round(total - float(ou), 1) if pd.notna(ou) else None
        flags = confidence_flags(r)

        # NaN is not valid JSON, so unavailable readings go out as null.
        def _j(v):
            return None if v is None or pd.isna(v) else round(float(v), 1)

        wx = {
            "temp_f": _j(r["temp_f"]), "wind_mph": _j(r["wind_mph"]),
            "precip_in": _j(r["precip_in"]), "humidity": _j(r["humidity"]),
            "is_dome": int(r["is_dome"]),
        }

        # --- comparable games, as context for the forecast ------------------
        assess = comps_cal.assess(r.get("comp_home_cover_rate"))
        comps = _comp_summary(r, assess)

        records.append({
            "game_id": None if pd.isna(r["game_id"]) else int(r["game_id"]),
            "kickoff_utc": None if pd.isna(r["kickoff"]) else r["kickoff"].isoformat(),
            "kickoff_et": None if pd.isna(r["kickoff"]) else
                          r["kickoff"].tz_convert(ET).strftime("%a %-I:%M %p ET"),
            "week": int(r["week"]),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "neutral_site": bool(r["neutral_site"]),

            # The forecast is the headline now.
            "forecast": {
                "home_points": round(float(r["pred_home_points"])),
                "away_points": round(float(r["pred_away_points"])),
                "margin": round(margin, 1),
                "total": round(total, 1),
                "home_win_prob": round(prob, 4),
            },
            "comps": comps,
            "comp_examples": (comps_engine.examples(r, k=5)
                              if comps_engine is not None
                              and comps_engine.available else []),
            "confidence_flags": flags,
            "vs_market": {"margin": raw_edge, "total": total_edge},

            # The market, for reference.
            "market_spread": None if market_margin is None else round(float(spread), 1),
            "market_total": None if pd.isna(ou) else round(float(ou), 1),

            "weather": wx,
            "weather_text": describe(wx),
            "games_played": int(min(r["home_played"], r["away_played"])),
        })

    # A preview tool reads like a schedule: earliest kickoff first.
    records.sort(key=lambda x: (x["kickoff_utc"] or "", x["home_team"]))

    metrics = {}
    if (config.MODELS / "metrics.json").exists():
        metrics = json.loads((config.MODELS / "metrics.json").read_text())

    return {
        "generated_at": datetime.now(ET).isoformat(),
        "season": season,
        "dates": [d.isoformat() for d in target_dates],
        "games": records,
        "model_metrics": metrics,
        "api_calls": client.calls_made,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predict today's college football slate")
    p.add_argument("--date", help="YYYY-MM-DD (default: today, US Eastern)")
    p.add_argument("--days", type=int, default=1,
                   help="how many days forward to include")
    p.add_argument("--no-weather", action="store_true",
                   help="skip the forecast entirely and predict immediately")
    p.add_argument("--weather-budget", type=float, default=120.0,
                   help="seconds to spend on weather before giving up (default 120)")
    p.add_argument("--out", default=str(config.DOCS / "predictions.json"))
    p.add_argument("--html", default=str(config.DOCS / "index.html"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
             else datetime.now(ET).date())
    targets = [start + timedelta(days=i) for i in range(max(1, args.days))]

    try:
        payload = run(targets, with_weather=not args.no_weather,
                      weather_budget=args.weather_budget)
    except MissingKeyError as exc:
        print(f"\n{exc}\n")
        return 2
    except FeatureMismatchError as exc:
        print(f"\n::error::{exc}\n")
        return 3

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    from dashboard import render
    html = render(payload)
    with open(args.html, "w") as fh:
        fh.write(html)

    n = len(payload["games"])
    plays = sum(1 for g in payload["games"] if g.get("spread_tier"))
    print(f"{n} games predicted, {plays} with a flagged edge.")
    print(f"JSON -> {args.out}")
    print(f"HTML -> {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
