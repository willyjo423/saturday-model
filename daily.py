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
from features import build_features
from ratings import RatingsEngine
from weather import WeatherService, describe

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MODEL_PATH = config.MODELS / "cfb_model.joblib"


def current_season(today: date) -> int:
    """College football seasons straddle the new year; bowls in January
    belong to the previous season."""
    return today.year if today.month >= 7 else today.year - 1


def tier_for(edge: float) -> str | None:
    for threshold, label in config.EDGE_TIERS:
        if abs(edge) >= threshold:
            return label
    return None


def run(target_dates: list[date], api_key: str | None = None) -> dict:
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

    priors = build_priors(client, [season], games)
    engine = RatingsEngine(games, priors)

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
    weather = WeatherService()
    feat = build_features(slate, engine, weather, with_weather=True, historical=False)

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
        edge = (margin - market_margin) if market_margin is not None else None
        total_edge = (total - float(ou)) if pd.notna(ou) else None

        wx = {
            "temp_f": r["temp_f"], "wind_mph": r["wind_mph"],
            "precip_in": r["precip_in"], "humidity": r["humidity"],
            "is_dome": int(r["is_dome"]),
        }

        favorite = r["home_team"] if margin > 0 else r["away_team"]
        records.append({
            "game_id": None if pd.isna(r["game_id"]) else int(r["game_id"]),
            "kickoff_utc": None if pd.isna(r["kickoff"]) else r["kickoff"].isoformat(),
            "kickoff_et": None if pd.isna(r["kickoff"]) else
                          r["kickoff"].tz_convert(ET).strftime("%a %-I:%M %p ET"),
            "week": int(r["week"]),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "neutral_site": bool(r["neutral_site"]),
            "pred_margin": round(margin, 1),
            "pred_total": round(total, 1),
            "pred_home_points": round(float(r["pred_home_points"]), 1),
            "pred_away_points": round(float(r["pred_away_points"]), 1),
            "home_win_prob": round(prob, 4),
            "favorite": favorite,
            "model_spread": round(-margin, 1),   # quoted home-team style
            "market_spread": None if market_margin is None else round(float(spread), 1),
            "market_total": None if pd.isna(ou) else round(float(ou), 1),
            "spread_edge": None if edge is None else round(edge, 1),
            "total_edge": None if total_edge is None else round(total_edge, 1),
            "spread_tier": None if edge is None else tier_for(edge),
            "total_tier": None if total_edge is None else tier_for(total_edge),
            "spread_play": None if edge is None else (
                f"{r['home_team']} {spread:+.1f}" if edge > 0
                else f"{r['away_team']} {-float(spread):+.1f}"),
            "total_play": None if total_edge is None else (
                f"Over {float(ou):.1f}" if total_edge > 0 else f"Under {float(ou):.1f}"),
            "weather": wx,
            "weather_text": describe(wx),
            "home_rating": round(float(r["home_rating"]), 1),
            "away_rating": round(float(r["away_rating"]), 1),
            "games_played": int(min(r["home_played"], r["away_played"])),
        })

    records.sort(key=lambda x: (-(abs(x["spread_edge"]) if x["spread_edge"] is not None else -1),
                                x["kickoff_utc"] or ""))

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
    p.add_argument("--out", default=str(config.DOCS / "predictions.json"))
    p.add_argument("--html", default=str(config.DOCS / "index.html"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
             else datetime.now(ET).date())
    targets = [start + timedelta(days=i) for i in range(max(1, args.days))]

    try:
        payload = run(targets)
    except MissingKeyError as exc:
        print(f"\n{exc}\n")
        return 2

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
