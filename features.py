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
from comps import COMP_FEATURES
from efficiency import METRICS as EFF_METRICS
from ratings import RatingsEngine
from weather import WeatherService

log = logging.getLogger(__name__)

# Efficiency enters as matchup *edges* - one side's offence against the other
# side's defence - rather than as four raw numbers per metric. That is the
# quantity that actually decides a game, and it keeps the feature count sane.
EFF_EDGE_COLUMNS = [f"edge_{m}_{side}"
                    for m in EFF_METRICS if m != "plays"
                    for side in ("home", "away")]

EFF_RAW_COLUMNS = ["home_off_ppa", "home_def_ppa", "away_off_ppa", "away_def_ppa"]

EFF_PACE_COLUMNS = ["pace_sum", "pace_diff", "eff_games"]

CONTEXT_COLUMNS = [
    "home_talent", "away_talent", "talent_diff",
    "home_returning", "away_returning",
    "home_returning_pass", "away_returning_pass",
]

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
    # weather. Left as NaN when genuinely unavailable - the booster handles
    # missing values natively, and a real gap must not masquerade as a mild
    # calm afternoon, which is what a filled-in default would teach it.
    "temp_f", "humidity", "precip_in", "wind_mph", "is_dome",
] + EFF_EDGE_COLUMNS + EFF_RAW_COLUMNS + EFF_PACE_COLUMNS + CONTEXT_COLUMNS \
  + COMP_FEATURES

TARGETS = ["margin", "total", "home_win"]


class SeasonContext:
    """Preseason facts about a team: recruiting talent and returning production.

    All of it is published before a snap is played, so using it at any point in
    the season is leak-free. Returning *passing* production is the closest
    thing CFBD offers to "did the starting quarterback come back", which is
    worth real points and is invisible to a margin-based rating.
    """

    def __init__(self):
        self.talent: dict[tuple[int, str], float] = {}
        self.returning: dict[tuple[int, str], float] = {}
        self.returning_pass: dict[tuple[int, str], float] = {}

    @staticmethod
    def _z(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        sd = s.std(ddof=0)
        if not sd or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / sd

    @staticmethod
    def _key_col(df: pd.DataFrame) -> str | None:
        for c in ("team", "school"):
            if c in df.columns:
                return c
        return None

    def add_season(self, year: int, talent: pd.DataFrame | None,
                   returning: pd.DataFrame | None) -> None:
        if talent is not None and not talent.empty:
            key = self._key_col(talent)
            if key and "talent" in talent.columns:
                z = self._z(talent["talent"])
                for team, val in zip(talent[key], z):
                    if pd.notna(val):
                        self.talent[(year, str(team).strip())] = float(val)

        if returning is not None and not returning.empty:
            key = self._key_col(returning)
            if key:
                total_col = next((c for c in ("totalPPA", "total_ppa", "percentPPA",
                                              "percent_ppa") if c in returning), None)
                pass_col = next((c for c in ("passingPPA", "totalPassingPPA",
                                             "percentPassingPPA", "passing_ppa")
                                 if c in returning), None)
                if total_col:
                    z = self._z(returning[total_col])
                    for team, val in zip(returning[key], z):
                        if pd.notna(val):
                            self.returning[(year, str(team).strip())] = float(val)
                if pass_col:
                    z = self._z(returning[pass_col])
                    for team, val in zip(returning[key], z):
                        if pd.notna(val):
                            self.returning_pass[(year, str(team).strip())] = float(val)

    def get(self, year: int, team: str) -> dict:
        return {
            "talent": self.talent.get((year, team), np.nan),
            "returning": self.returning.get((year, team), np.nan),
            "returning_pass": self.returning_pass.get((year, team), np.nan),
        }


def _num(x, default=np.nan) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _efficiency_features(eff_h: dict, eff_a: dict) -> dict:
    """Matchup edges: one side's offence against the other side's defence.

    The efficiency solve is additive - an observed offensive rate is that
    offence plus the defence it faced - so the expected rate for this matchup
    is the sum of the two, not the difference.
    """
    out: dict[str, float] = {}
    for m in EFF_METRICS:
        if m == "plays":
            continue
        out[f"edge_{m}_home"] = eff_h[f"off_{m}"] + eff_a[f"def_{m}"]
        out[f"edge_{m}_away"] = eff_a[f"off_{m}"] + eff_h[f"def_{m}"]

    out["home_off_ppa"] = eff_h["off_ppa"]
    out["home_def_ppa"] = eff_h["def_ppa"]
    out["away_off_ppa"] = eff_a["off_ppa"]
    out["away_def_ppa"] = eff_a["def_ppa"]

    # Tempo. A game total is mostly a question of how many snaps get run.
    home_plays = eff_h["off_plays"] + eff_a["def_plays"]
    away_plays = eff_a["off_plays"] + eff_h["def_plays"]
    out["pace_sum"] = home_plays + away_plays
    out["pace_diff"] = home_plays - away_plays
    out["eff_games"] = min(eff_h.get("eff_games", 0.0), eff_a.get("eff_games", 0.0))
    return out


_BLANK_EFF = {f"{side}_{m}": np.nan
              for m in EFF_METRICS for side in ("off", "def")}
_BLANK_EFF["eff_games"] = 0.0


def build_features(games: pd.DataFrame, engine: RatingsEngine,
                   weather: WeatherService | None = None,
                   with_weather: bool = True,
                   historical: bool | None = None,
                   efficiency=None,
                   context: SeasonContext | None = None) -> pd.DataFrame:
    """Turn a game table into a model-ready feature matrix."""
    rows = []
    for _, g in games.iterrows():
        season, week = int(g["season"]), int(g["week"])
        h = engine.lookup(season, week, g["home_team"])
        a = engine.lookup(season, week, g["away_team"])

        if efficiency is not None:
            eff_h = efficiency.lookup(season, week, g["home_team"])
            eff_a = efficiency.lookup(season, week, g["away_team"])
        else:
            eff_h = eff_a = dict(_BLANK_EFF)
        eff = _efficiency_features(eff_h, eff_a)

        if context is not None:
            ctx_h = context.get(season, g["home_team"])
            ctx_a = context.get(season, g["away_team"])
        else:
            ctx_h = ctx_a = {"talent": np.nan, "returning": np.nan,
                             "returning_pass": np.nan}

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

            "temp_f": _num(wx.get("temp_f")),
            "humidity": _num(wx.get("humidity")),
            "precip_in": _num(wx.get("precip_in")),
            "wind_mph": _num(wx.get("wind_mph")),
            "is_dome": int(wx.get("is_dome", 0) or 0),

            "home_talent": ctx_h["talent"], "away_talent": ctx_a["talent"],
            "home_returning": ctx_h["returning"],
            "away_returning": ctx_a["returning"],
            "home_returning_pass": ctx_h["returning_pass"],
            "away_returning_pass": ctx_a["returning_pass"],

            "margin": _num(g.get("margin")),
            "total": _num(g.get("total")),
            "spread": _num(g.get("spread")),
            "over_under": _num(g.get("over_under")),
        }
        row.update(eff)
        row["talent_diff"] = row["home_talent"] - row["away_talent"]
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
