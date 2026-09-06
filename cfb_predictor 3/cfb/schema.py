"""Column normalisation.

CFBD has shipped both snake_case and camelCase field names across API versions,
and `json_normalize` flattens nested objects with dots. Rather than pin to one
spelling and break on the next release, every consumer goes through here.
"""
from __future__ import annotations

import pandas as pd

GAME_FIELDS = {
    "game_id": ["id", "gameId", "game_id"],
    "season": ["season", "year"],
    "week": ["week"],
    "season_type": ["seasonType", "season_type"],
    "start_date": ["startDate", "start_date", "startTimeTbd.startDate"],
    "start_time_tbd": ["startTimeTbd", "start_time_tbd"],
    "neutral_site": ["neutralSite", "neutral_site"],
    "conference_game": ["conferenceGame", "conference_game"],
    "venue_id": ["venueId", "venue_id"],
    "venue": ["venue", "venue.name"],
    "home_id": ["homeId", "home_id", "homeTeam.id"],
    "home_team": ["homeTeam", "home_team", "homeTeam.name"],
    "home_conference": ["homeConference", "home_conference"],
    "home_points": ["homePoints", "home_points"],
    "home_classification": ["homeClassification", "home_classification",
                            "homeDivision", "home_division"],
    "away_id": ["awayId", "away_id", "awayTeam.id"],
    "away_team": ["awayTeam", "away_team", "awayTeam.name"],
    "away_conference": ["awayConference", "away_conference"],
    "away_points": ["awayPoints", "away_points"],
    "away_classification": ["awayClassification", "away_classification",
                            "awayDivision", "away_division"],
    "completed": ["completed"],
}

VENUE_FIELDS = {
    "venue_id": ["id", "venueId"],
    "venue": ["name"],
    "city": ["city"],
    "state": ["state"],
    "timezone": ["timezone"],
    "lat": ["latitude", "location.y", "location.latitude"],
    "lon": ["longitude", "location.x", "location.longitude"],
    "elevation": ["elevation", "location.elevation"],
    "capacity": ["capacity"],
    "grass": ["grass"],
    "dome": ["dome", "indoor"],
}

LINE_FIELDS = {
    "game_id": ["id", "gameId", "game_id"],
    "season": ["season", "year"],
    "week": ["week"],
    "home_team": ["homeTeam", "home_team"],
    "away_team": ["awayTeam", "away_team"],
    "lines": ["lines"],
}


# Fields the pipeline genuinely cannot run without. Everything else in the
# mappings above is enrichment that degrades to NA if an endpoint changes.
REQUIRED_GAME_FIELDS = [
    "game_id", "season", "week", "start_date", "neutral_site",
    "home_team", "away_team", "home_points", "away_points", "venue_id",
]
REQUIRED_VENUE_FIELDS = ["venue_id", "lat", "lon"]


def _first_present(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def normalise(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    """Project a raw API frame onto canonical column names.

    Missing fields become all-NA columns rather than raising, so a change in
    one endpoint degrades that feature instead of killing the run.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(mapping))
    out = {}
    for canonical, candidates in mapping.items():
        series = _first_present(df, candidates)
        out[canonical] = series if series is not None else pd.Series(
            [pd.NA] * len(df), index=df.index)
    return pd.DataFrame(out)


def missing_report(df: pd.DataFrame, mapping: dict[str, list[str]]) -> list[str]:
    """Which canonical fields the API did not supply. Used by the self-test."""
    if df is None or df.empty:
        return list(mapping)
    return [c for c, cands in mapping.items()
            if _first_present(df, cands) is None]
