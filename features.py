"""Feature construction.

Every feature here is computable *before* kickoff. The two rules that keep
this honest:

* Team strength comes from `RatingsEngine.ratings_before(season, week)`, which
  by construction sees no game from `week` or later.
* Betting lines are deliberately **not** features. If the model trained on the
  closing spread it would mostly learn to reproduce it, and "model vs market"
  would stop meaning anything. Lines are used only for evaluation and for
  computing edge at prediction time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from ratings import RatingsEngine
from weather import WeatherService

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    # strength
    "rating_diff", "rating_sum",
    "home_rating", "away_rating",
    "home_off", "home_def", "away_off", "away_def",
    "proj_margin", "proj_total",
    "off_matchup_home", "off_matchup_away",
    # confidence in those ratings
    "home_played", "away_played", "min_played",
    "home_known", "away_known",
    # situation
    "week", "neutral_site", "conference_game",
    "home_rest", "away_rest", "rest_diff",
    "home_game_no", "away_game_no",
    "away_travel_mi", "away_alt_change", "elevation",
    # weather
    "temp_f", "humidity", "precip_in", "wind_mph", "gust_mph", "is_dome",
]

TARGETS = ["margin", "total", "home_win"]


def _num(x, default=np.nan) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def build_features(games: pd.DataFrame, engine: RatingsEngine,
                   weather: WeatherService | None = None,
                   with_weather: bool = True,
                   historical: bool | None = None) -> pd.DataFrame:
    """Turn a game table into a model-ready feature matrix."""
    rows = []
    for _, g in games.iterrows():
        season, week = int(g["season"]), int(g["week"])
        h = engine.lookup(season, week, g["home_team"])
        a = engine.lookup(season, week, g["away_team"])

        neutral = bool(g["neutral_site"])
        hfa = 0.0 if neutral else h["hfa"]

        proj_home_pts = h["off"] + a["def"] + hfa / 2
        proj_away_pts = a["off"] + h["def"] - hfa / 2

        wx = {}
        if with_weather and weather is not None:
            wx = weather.at_kickoff(
                g.get("lat"), g.get("lon"), g.get("kickoff"),
                is_dome=bool(g.get("dome", False)),
                historical=historical,
            )

        row = {
            "game_id": g.get("game_id"),
            "season": season,
            "week": week,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "kickoff": g.get("kickoff"),

            "home_rating": h["rating"],
            "away_rating": a["rating"],
            "rating_diff": h["rating"] - a["rating"],
            "rating_sum": h["rating"] + a["rating"],
            "home_off": h["off"], "home_def": h["def"],
            "away_off": a["off"], "away_def": a["def"],
            "proj_margin": (h["rating"] - a["rating"]) + hfa,
            "proj_total": proj_home_pts + proj_away_pts,
            "off_matchup_home": h["off"] - a["def"],
            "off_matchup_away": a["off"] - h["def"],
            "home_played": h["played"], "away_played": a["played"],
            "min_played": min(h["played"], a["played"]),
            "home_known": h["known"], "away_known": a["known"],

            "neutral_site": int(neutral),
            "conference_game": int(bool(g.get("conference_game", False))),
            "home_rest": _num(g.get("home_rest_days"), 14.0),
            "away_rest": _num(g.get("away_rest_days"), 14.0),
            "home_game_no": _num(g.get("home_game_no"), 0.0),
            "away_game_no": _num(g.get("away_game_no"), 0.0),
            "away_travel_mi": _num(g.get("away_travel_mi"), 500.0),
            "away_alt_change": _num(g.get("away_alt_change"), 0.0),
            "elevation": _num(g.get("elevation"), 500.0),

            "temp_f": _num(wx.get("temp_f"), 65.0),
            "humidity": _num(wx.get("humidity"), 60.0),
            "precip_in": _num(wx.get("precip_in"), 0.0),
            "wind_mph": _num(wx.get("wind_mph"), 6.0),
            "gust_mph": _num(wx.get("gust_mph"), 10.0),
            "is_dome": int(wx.get("is_dome", 0)),

            "margin": _num(g.get("margin")),
            "total": _num(g.get("total")),
            "spread": _num(g.get("spread")),
            "over_under": _num(g.get("over_under")),
        }
        row["rest_diff"] = row["home_rest"] - row["away_rest"]
        row["home_win"] = (
            1 if row["margin"] > 0 else (0 if row["margin"] < 0 else np.nan)
        ) if not np.isnan(row["margin"]) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def training_matrix(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into X and y, dropping games we can't learn from."""
    usable = feat["margin"].notna() & feat["total"].notna()
    # Games where neither side has a real rating teach the model nothing.
    usable &= (feat["home_known"] == 1) | (feat["away_known"] == 1)
    df = feat.loc[usable].copy()
    return df[FEATURE_COLUMNS], df[TARGETS + ["season", "week", "spread", "over_under"]]
