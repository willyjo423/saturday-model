"""Assemble the canonical game table and its context (venues, lines, rest, travel)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config, schema
from api import CFBDClient, CFBDError

log = logging.getLogger(__name__)

EARTH_RADIUS_MI = 3958.8


def haversine(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    p1, p2 = np.radians(float(lat1)), np.radians(float(lat2))
    dphi = p2 - p1
    dlam = np.radians(float(lon2) - float(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a)))


def load_venues(client: CFBDClient) -> pd.DataFrame:
    raw = client.venues()
    v = schema.normalise(raw, schema.VENUE_FIELDS)
    for col in ("lat", "lon", "elevation", "capacity"):
        v[col] = pd.to_numeric(v[col], errors="coerce")
    v["dome"] = v["dome"].fillna(False).astype(bool)
    v["venue_id"] = pd.to_numeric(v["venue_id"], errors="coerce")
    return v.dropna(subset=["venue_id"]).drop_duplicates("venue_id")


def load_games(client: CFBDClient, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        try:
            raw = client.all_games(year)
        except CFBDError as exc:
            log.error("Could not load %s games: %s", year, exc)
            continue
        if raw.empty:
            continue
        g = schema.normalise(raw, schema.GAME_FIELDS)
        g["season"] = pd.to_numeric(g["season"], errors="coerce").fillna(year)
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=list(schema.GAME_FIELDS))
    return pd.concat(frames, ignore_index=True)


def clean_games(games: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()

    for col in ("season", "week", "home_points", "away_points",
                "venue_id", "home_id", "away_id"):
        g[col] = pd.to_numeric(g[col], errors="coerce")

    g["neutral_site"] = g["neutral_site"].fillna(False).astype(bool)
    g["conference_game"] = g["conference_game"].fillna(False).astype(bool)
    g["kickoff"] = pd.to_datetime(g["start_date"], utc=True, errors="coerce")
    g["home_team"] = g["home_team"].astype(str).str.strip()
    g["away_team"] = g["away_team"].astype(str).str.strip()
    g = g.dropna(subset=["season", "week"])
    g["season"] = g["season"].astype(int)
    g["week"] = g["week"].astype(int)

    # Postseason games get pushed past the regular season so week ordering
    # stays monotonic for the ratings engine.
    post = g["season_type"].astype(str).str.lower().eq("postseason")
    g.loc[post, "week"] = config.REGULAR_SEASON_WEEKS + 2

    # Pool non-FBS opponents so blowouts over FCS teams don't mint ratings.
    from ratings import FCS
    for side in ("home", "away"):
        cls = g[f"{side}_classification"].astype(str).str.lower()
        is_fbs = cls.eq("fbs") | cls.eq("nan") | cls.eq("<na>")
        g[f"{side}_is_fbs"] = is_fbs
        g.loc[~is_fbs, f"{side}_team"] = FCS

    g = g.merge(
        venues[["venue_id", "lat", "lon", "elevation", "dome", "capacity"]],
        on="venue_id", how="left")

    g["margin"] = g["home_points"] - g["away_points"]
    g["total"] = g["home_points"] + g["away_points"]
    g["completed"] = g["home_points"].notna() & g["away_points"].notna()

    return g.sort_values(["season", "week", "kickoff"]).reset_index(drop=True)


def home_venue_map(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's usual home coordinates, for computing travel distance."""
    home = games.loc[
        ~games["neutral_site"] & games["lat"].notna(),
        ["home_team", "season", "lat", "lon", "elevation"]
    ]
    agg = (home.groupby(["home_team", "season"])
                .agg(lat=("lat", "median"), lon=("lon", "median"),
                     elevation=("elevation", "median"))
                .reset_index()
                .rename(columns={"home_team": "team"}))
    return agg


def add_rest_and_travel(games: pd.DataFrame) -> pd.DataFrame:
    """Days of rest for each side, and how far the away team travelled.

    Rest uses only prior kickoffs, so it is safe to compute across the whole
    frame at once.
    """
    g = games.copy()
    hv = home_venue_map(g)
    hv_idx = hv.set_index(["team", "season"])

    # --- rest ---
    long = pd.concat([
        g[["game_id", "season", "kickoff", "home_team"]]
            .rename(columns={"home_team": "team"}).assign(side="home"),
        g[["game_id", "season", "kickoff", "away_team"]]
            .rename(columns={"away_team": "team"}).assign(side="away"),
    ], ignore_index=True)

    long = long.sort_values(["team", "season", "kickoff"])
    long["prev"] = long.groupby(["team", "season"])["kickoff"].shift(1)
    long["rest_days"] = (long["kickoff"] - long["prev"]).dt.total_seconds() / 86400
    long["rest_days"] = long["rest_days"].fillna(14.0).clip(0, 30)
    long["game_no"] = long.groupby(["team", "season"]).cumcount()

    rest = long.pivot_table(index="game_id", columns="side",
                            values=["rest_days", "game_no"], aggfunc="first")
    rest.columns = [f"{b}_{a}" for a, b in rest.columns]
    g = g.merge(rest, left_on="game_id", right_index=True, how="left")

    # --- travel + altitude change ---
    def _travel(row):
        if row["neutral_site"]:
            keys = [("home", row["home_team"]), ("away", row["away_team"])]
        else:
            keys = [("away", row["away_team"])]
        out = {"away_travel_mi": np.nan, "home_travel_mi": 0.0,
               "away_alt_change": 0.0}
        for side, team in keys:
            try:
                base = hv_idx.loc[(team, row["season"])]
            except KeyError:
                continue
            dist = haversine(base["lat"], base["lon"], row["lat"], row["lon"])
            out[f"{side}_travel_mi"] = dist
            if side == "away" and pd.notna(row["elevation"]) and pd.notna(base["elevation"]):
                out["away_alt_change"] = float(row["elevation"]) - float(base["elevation"])
        return pd.Series(out)

    travel = g.apply(_travel, axis=1)
    g = pd.concat([g, travel], axis=1)
    g["away_travel_mi"] = g["away_travel_mi"].fillna(g["away_travel_mi"].median())
    g["home_travel_mi"] = g["home_travel_mi"].fillna(0.0)
    return g


BLANK_LINE = {
    "spread": np.nan, "over_under": np.nan, "provider": None,
    "n_providers": 0.0, "spread_dispersion": np.nan,
    "spread_open": np.nan, "spread_move": np.nan,
    "total_dispersion": np.nan,
}


def _best_line(entries) -> dict:
    """Consensus spread and total, plus what the *disagreement* between books says.

    Taking a median and discarding the rest throws away the most useful thing
    in this payload. Two numbers matter beyond the consensus:

    * **dispersion** - how far apart the books are. Tight agreement means the
      market is confident, so a large model disagreement is far more likely to
      be our error than an opportunity. Wide dispersion means the market itself
      is unsure.
    * **movement** - where the line opened versus where it sits now. Money
      moving toward our number is corroboration; moving away means the market
      learned something we haven't.

    Neither is a prediction input. Both feed confidence, which is a different
    question, and keeping them out of the model preserves the honesty of the
    model-versus-market comparison.
    """
    if not isinstance(entries, list) or not entries:
        return dict(BLANK_LINE)

    spreads, totals, opens, providers = [], [], [], []
    preferred = {"consensus", "draftkings", "bovada", "espn bet", "teamrankings"}

    for e in entries:
        if not isinstance(e, dict):
            continue
        sp = e.get("spread")
        ou = e.get("overUnder", e.get("over_under"))
        op = e.get("spreadOpen", e.get("spread_open"))
        providers.append(str(e.get("provider", "")).lower())
        if sp is not None:
            try:
                spreads.append(float(sp))
            except (TypeError, ValueError):
                pass
        if ou is not None:
            try:
                totals.append(float(ou))
            except (TypeError, ValueError):
                pass
        if op is not None:
            try:
                opens.append(float(op))
            except (TypeError, ValueError):
                pass

    if not spreads and not totals:
        return dict(BLANK_LINE)

    pick = next((p for p in providers if p in preferred),
                providers[0] if providers else None)
    consensus = float(np.median(spreads)) if spreads else np.nan
    open_line = float(np.median(opens)) if opens else np.nan

    return {
        "spread": consensus,
        "over_under": float(np.median(totals)) if totals else np.nan,
        "provider": pick,
        "n_providers": float(len(spreads)),
        # Max-minus-min rather than sd: with two or three books, the range is
        # the honest description of how far apart they are.
        "spread_dispersion": (float(max(spreads) - min(spreads))
                              if len(spreads) > 1 else 0.0),
        "spread_open": open_line,
        "spread_move": (consensus - open_line
                        if not np.isnan(consensus) and not np.isnan(open_line)
                        else np.nan),
        "total_dispersion": (float(max(totals) - min(totals))
                             if len(totals) > 1 else 0.0),
    }


def load_lines(client: CFBDClient, years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        for st in ("regular", "postseason"):
            try:
                raw = client.lines(year, season_type=st)
            except CFBDError as exc:
                log.warning("lines %s %s: %s", year, st, exc)
                continue
            if raw.empty:
                continue
            norm = schema.normalise(raw, schema.LINE_FIELDS)
            for _, r in norm.iterrows():
                best = _best_line(r["lines"])
                rows.append({"game_id": r["game_id"], **best})
    if not rows:
        return pd.DataFrame(columns=["game_id"] + list(BLANK_LINE))
    df = pd.DataFrame(rows)
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df = df.dropna(subset=["game_id"]).drop_duplicates("game_id")
    if "spread_open" in df:
        has_open = df["spread_open"].notna().mean()
        log.info("lines: %d games, opening number available for %.0f%%",
                 len(df), has_open * 100)
    return df
