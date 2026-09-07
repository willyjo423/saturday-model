"""Opponent-adjusted play-level efficiency, computed week by week.

Why this exists
---------------
Scoring margin alone cannot tell a good team from a lucky one. A team that
wins by 10 after being outgained, on the back of two fumble recoveries, looks
identical to one that won by 10 while dominating every phase - and the second
team is far more likely to win next week. Play-level efficiency separates
them, and it stabilises much faster: a usable read by week 4 rather than week 8.

What is measured
----------------
Per team, per side of the ball, opponent-adjusted:

* **ppa**          expected points added per play - the headline efficiency number
* **success_rate** share of plays that stayed on schedule
* **explosiveness** average value of the plays that did succeed
* **line_yards**   yards credited to the offensive line - trench control
* **stuff_rate**   share of runs stopped at or behind the line
* **power_success** short-yardage conversion rate
* **havoc**        share of plays disrupted by the defence
* **plays**        tempo, which is most of what a game total actually is

Leakage
-------
Everything routes through `stats_before(year, week)`, which is built only from
games completed before that week - the same guarantee the power ratings carry,
and it is tested the same way.

Efficiency
----------
All metrics share one design matrix (team offence and defence indicators), so a
single ridge factorisation per week solves every metric at once via multiple
right-hand sides, rather than one solve per metric.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from ratings import FCS

log = logging.getLogger(__name__)

# canonical name -> candidate field names as the API may spell them
OFFENSE_FIELDS = {
    "ppa": ["offense.ppa", "offense.PPA"],
    "success_rate": ["offense.successRate", "offense.success_rate"],
    "explosiveness": ["offense.explosiveness"],
    "line_yards": ["offense.lineYards", "offense.line_yards"],
    "stuff_rate": ["offense.stuffRate", "offense.stuff_rate"],
    "power_success": ["offense.powerSuccess", "offense.power_success"],
    "plays": ["offense.plays"],
    "havoc": ["offense.havoc.total", "offense.havocTotal"],
}
DEFENSE_FIELDS = {
    "ppa": ["defense.ppa", "defense.PPA"],
    "success_rate": ["defense.successRate", "defense.success_rate"],
    "explosiveness": ["defense.explosiveness"],
    "line_yards": ["defense.lineYards", "defense.line_yards"],
    "stuff_rate": ["defense.stuffRate", "defense.stuff_rate"],
    "power_success": ["defense.powerSuccess", "defense.power_success"],
    "plays": ["defense.plays"],
    "havoc": ["defense.havoc.total", "defense.havocTotal"],
}

METRICS = list(OFFENSE_FIELDS)

IDENTITY_FIELDS = {
    "game_id": ["gameId", "game_id", "id"],
    "season": ["season", "year"],
    "week": ["week"],
    "team": ["team"],
    "opponent": ["opponent"],
}

# Ridge strength for the efficiency solve. Rates are noisier per game than
# scoring margin, so they shrink harder toward the league mean.
EFF_LAMBDA = 14.0


def _pick(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def normalise_game_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten CFBD advanced game stats into one row per team-game.

    Fields the endpoint does not supply become all-NA columns, so a change on
    their side costs us that one metric rather than the whole feature group.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = {}
    for canon, cands in IDENTITY_FIELDS.items():
        found = None
        for c in cands:
            if c in raw.columns:
                found = raw[c]
                break
        out[canon] = found if found is not None else pd.Series([pd.NA] * len(raw))

    missing = []
    for metric in METRICS:
        off = _pick(raw, OFFENSE_FIELDS[metric])
        dfn = _pick(raw, DEFENSE_FIELDS[metric])
        if off is None and dfn is None:
            missing.append(metric)
        out[f"off_{metric}"] = off if off is not None else np.nan
        out[f"def_{metric}"] = dfn if dfn is not None else np.nan

    if missing:
        log.warning("advanced stats missing these metrics entirely: %s", missing)

    df = pd.DataFrame(out)
    for col in ("season", "week", "game_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team"] = df["team"].astype(str).str.strip()
    df["opponent"] = df["opponent"].astype(str).str.strip()
    return df.dropna(subset=["season", "week"])


class EfficiencyEngine:
    """Opponent-adjusted efficiency at any point in a season.

    Mirrors RatingsEngine: `stats_before(year, week)` uses only games played
    before that week, and results are cached per (year, week).
    """

    def __init__(self, team_games: pd.DataFrame, fbs_only: bool = True):
        self.data = team_games if team_games is not None else pd.DataFrame()
        self.fbs_only = fbs_only
        self._cache: dict[tuple[int, int], pd.DataFrame] = {}
        self.available = not self.data.empty
        if not self.available:
            log.warning("no advanced game stats available - efficiency "
                        "features will be NaN and the model will lean on "
                        "scoring margin alone")

    # -- solving ------------------------------------------------------------
    def _solve(self, played: pd.DataFrame) -> pd.DataFrame:
        """One ridge factorisation, every metric solved as a separate RHS.

        For each team-game row: observed = offence(team) + defence(opponent),
        which separates a team's own quality from the standard of opposition
        it has faced. Identical in shape to the points solve in ratings.py.
        """
        teams = sorted(set(played["team"]) | set(played["opponent"]))
        if len(teams) < 4:
            return pd.DataFrame()

        idx = {t: i for i, t in enumerate(teams)}
        n_t, n_r = len(teams), len(played)

        X = np.zeros((n_r, 2 * n_t))
        rows = np.arange(n_r)
        X[rows, played["team"].map(idx).to_numpy()] = 1.0
        X[rows, n_t + played["opponent"].map(idx).to_numpy()] = 1.0

        # Offensive metrics use the team's own offence vs the opponent's
        # defence; defensive metrics are the mirror image, so we stack both
        # into one system by flipping which side owns the observation.
        y_cols, kept = [], []
        for metric in METRICS:
            col = played[f"off_{metric}"]
            if col.notna().sum() < max(30, 0.2 * n_r):
                continue
            y_cols.append(col.fillna(col.mean()).to_numpy(dtype=float))
            kept.append(metric)

        if not kept:
            return pd.DataFrame()

        Y = np.column_stack(y_cols)
        means = Y.mean(axis=0)

        P = np.eye(2 * n_t) * EFF_LAMBDA
        # Anchor each half of the system so offence and defence are identified.
        anchor = np.zeros((2, 2 * n_t))
        anchor[0, :n_t] = 1.0
        anchor[1, n_t:] = 1.0
        Xa = np.vstack([X, anchor * 5.0])
        Ya = np.vstack([Y - means, np.zeros((2, len(kept)))])

        A = Xa.T @ Xa + P
        beta = np.linalg.solve(A, Xa.T @ Ya)

        counts = played["team"].value_counts()
        result = {"team": teams,
                  "eff_games": [float(counts.get(t, 0)) for t in teams]}
        for j, metric in enumerate(kept):
            result[f"off_{metric}"] = beta[:n_t, j]
            result[f"def_{metric}"] = beta[n_t:2 * n_t, j]

        out = pd.DataFrame(result)
        out.attrs["metrics"] = kept
        out.attrs["means"] = dict(zip(kept, means))
        return out

    # -- public API ---------------------------------------------------------
    def stats_before(self, year: int, week: int) -> pd.DataFrame:
        key = (year, week)
        if key in self._cache:
            return self._cache[key]
        if not self.available:
            self._cache[key] = pd.DataFrame()
            return self._cache[key]

        d = self.data
        mask = (d["season"] == year) & (d["week"] < week)
        played = d.loc[mask]
        result = self._solve(played) if len(played) >= 20 else pd.DataFrame()
        self._cache[key] = result
        return result

    def lookup(self, year: int, week: int, team: str) -> dict:
        """Efficiency for one team, or NaNs when there is nothing to report."""
        blank = {f"{side}_{m}": np.nan
                 for m in METRICS for side in ("off", "def")}
        blank["eff_games"] = 0.0

        table = self.stats_before(year, week)
        if table.empty:
            return blank

        row = table.loc[table["team"] == team]
        if row.empty:
            # An unrated team (FCS, or yet to play) sits at the league mean by
            # construction, since the solve is centred - so zeros, not NaN.
            out = {k: (0.0 if k != "eff_games" else 0.0) for k in blank}
            return out

        r = row.iloc[0]
        out = dict(blank)
        for m in table.attrs.get("metrics", []):
            out[f"off_{m}"] = float(r[f"off_{m}"])
            out[f"def_{m}"] = float(r[f"def_{m}"])
        out["eff_games"] = float(r["eff_games"])
        return out


def load_team_game_stats(client, years: list[int]) -> pd.DataFrame:
    """Fetch advanced per-game stats for the given seasons.

    One call per season per season-type. If the endpoint is unavailable on the
    caller's CFBD tier this returns empty and the pipeline carries on without
    efficiency features rather than failing.
    """
    from api import CFBDError

    frames = []
    for year in years:
        for st in ("regular", "postseason"):
            try:
                raw = client.advanced_game_stats(year, season_type=st)
            except CFBDError as exc:
                log.warning("advanced game stats %s %s unavailable: %s",
                            year, st, exc)
                continue
            if raw is None or raw.empty:
                continue
            norm = normalise_game_stats(raw)
            if norm.empty:
                continue
            norm["season"] = norm["season"].fillna(year)
            frames.append(norm)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    log.info("advanced game stats: %d team-games, %s-%s",
             len(df), df["season"].min(), df["season"].max())
    return df


def pool_non_fbs(stats: pd.DataFrame, fbs_teams: set[str]) -> pd.DataFrame:
    """Map non-FBS opponents onto the pooled FCS team, as the ratings do."""
    if stats.empty or not fbs_teams:
        return stats
    out = stats.copy()
    out.loc[~out["team"].isin(fbs_teams), "team"] = FCS
    out.loc[~out["opponent"].isin(fbs_teams), "opponent"] = FCS
    return out
