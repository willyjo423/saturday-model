"""Leak-free power ratings.

The single most important guarantee in this project: a rating used to predict
a game in week W of season Y is fit **only** on games played before week W of
season Y, plus a preseason prior built from information known before the
season started (prior-year SP+, recruiting talent, returning production).

Method
------
Ridge-regularised least squares on capped scoring margin:

    margin(home, away) = r_home - r_away + hfa * (not neutral)

minimising ||Xb - y||^2 + lambda * ||b - prior||^2. The prior term is what
makes week 1 sane: with no games played, every rating collapses to its
preseason prior, and as games accumulate the data takes over smoothly. No
special-casing of early weeks needed.

A second solve splits the same games into offence and defence ratings, which
is what gives the totals model something to work with.

Non-FBS opponents are pooled into a single synthetic team so that a 56-0 win
over an FCS school doesn't buy a real rating.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

FCS = "__FCS__"


def _cap(series: pd.Series, cap: float) -> pd.Series:
    """Soft-cap margins. Blowouts carry information but not linearly."""
    s = series.astype(float)
    sign = np.sign(s)
    mag = s.abs()
    over = mag > cap
    mag = mag.where(~over, cap + np.sqrt(np.clip(mag - cap, 0, None)) * 2.0)
    return sign * mag


class PreseasonPriors:
    """Preseason rating priors, all from information available before kickoff."""

    def __init__(self):
        self.by_year: dict[int, pd.Series] = {}
        self.global_mean = 0.0

    @staticmethod
    def _z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        sd = s.std(ddof=0)
        if not sd or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    def build(self, year: int, prior_sp: pd.DataFrame | None,
              talent: pd.DataFrame | None, returning: pd.DataFrame | None,
              prior_final: pd.Series | None) -> pd.Series:
        """Blend last year's finish, recruiting talent, and returning production.

        Everything is expressed on the ratings scale (points vs. average).
        """
        components: list[pd.Series] = []
        weights: list[float] = []

        if prior_final is not None and len(prior_final):
            components.append(prior_final.astype(float) * config.YEAR_CARRYOVER)
            weights.append(1.0)

        if prior_sp is not None and not prior_sp.empty and "rating" in prior_sp:
            sp = prior_sp.set_index("team")["rating"]
            components.append(pd.to_numeric(sp, errors="coerce")
                              * config.YEAR_CARRYOVER)
            weights.append(0.8)

        if talent is not None and not talent.empty and "talent" in talent:
            tal = talent.set_index("school" if "school" in talent else "team")["talent"]
            components.append(self._z(tal) * 5.0)
            weights.append(0.5)

        if returning is not None and not returning.empty:
            col = next((c for c in ("totalPPA", "total_ppa", "percentPPA")
                        if c in returning), None)
            key = "team" if "team" in returning else "school"
            if col and key in returning:
                ret = returning.set_index(key)[col]
                components.append(self._z(ret) * 1.5)
                weights.append(0.3)

        if not components:
            self.by_year[year] = pd.Series(dtype=float)
            return self.by_year[year]

        idx = components[0].index
        for c in components[1:]:
            idx = idx.union(c.index)

        num = pd.Series(0.0, index=idx)
        den = pd.Series(0.0, index=idx)
        for comp, w in zip(components, weights):
            aligned = comp.reindex(idx)
            mask = aligned.notna()
            num[mask] += aligned[mask] * w
            den[mask] += w

        prior = (num / den.replace(0, np.nan)).fillna(0.0)
        # Centre so the prior describes "points better than an average FBS team".
        prior = prior - prior.mean()
        prior[FCS] = float(prior.min() - 12.0) if len(prior) else -25.0
        self.by_year[year] = prior
        return prior

    def get(self, year: int, teams: list[str]) -> np.ndarray:
        prior = self.by_year.get(year, pd.Series(dtype=float))
        return prior.reindex(teams).fillna(0.0).to_numpy(dtype=float)


class RatingsEngine:
    """Fits margin and offence/defence ratings at any point in a season."""

    def __init__(self, games: pd.DataFrame, priors: PreseasonPriors,
                 ridge_lambda: float | None = None):
        self.games = games
        self.priors = priors
        self.ridge_lambda = ridge_lambda or config.RIDGE_LAMBDA_BASE
        self._cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._final_cache: dict[int, pd.Series] = {}

    # -- solving ------------------------------------------------------------
    def _solve(self, played: pd.DataFrame, year: int) -> pd.DataFrame:
        teams = sorted(set(played["home_team"]) | set(played["away_team"]))
        if not teams:
            return pd.DataFrame(columns=["team", "rating", "off", "def", "played"])

        idx = {t: i for i, t in enumerate(teams)}
        n_t, n_g = len(teams), len(played)

        prior = self.priors.get(year, teams)

        # --- margin solve ---
        X = np.zeros((n_g, n_t + 1))
        rows = np.arange(n_g)
        X[rows, played["home_team"].map(idx).to_numpy()] = 1.0
        X[rows, played["away_team"].map(idx).to_numpy()] = -1.0
        X[:, n_t] = np.where(played["neutral_site"].to_numpy(), 0.0, 1.0)
        y = _cap(played["home_points"] - played["away_points"], config.MARGIN_CAP).to_numpy()

        lam = self.ridge_lambda
        P = np.eye(n_t + 1) * lam
        P[n_t, n_t] = lam * 0.25  # let HFA move more freely
        b0 = np.append(prior, config.HFA_PRIOR)

        # Anchor the mean rating at zero so the system is identifiable.
        anchor = np.zeros((1, n_t + 1))
        anchor[0, :n_t] = 1.0
        Xa = np.vstack([X, anchor * 10.0])
        ya = np.append(y, 0.0)

        A = Xa.T @ Xa + P
        rhs = Xa.T @ ya + P @ b0
        beta = np.linalg.solve(A, rhs)
        rating = beta[:n_t]
        hfa = float(beta[n_t])

        # --- offence / defence solve ---
        # Each game contributes two rows: points scored by each side.
        X2 = np.zeros((2 * n_g, 2 * n_t + 1))
        h = played["home_team"].map(idx).to_numpy()
        a = played["away_team"].map(idx).to_numpy()
        neutral = played["neutral_site"].to_numpy()

        r1 = np.arange(n_g)
        X2[r1, h] = 1.0                    # home offence
        X2[r1, n_t + a] = 1.0              # away defence
        X2[r1, 2 * n_t] = np.where(neutral, 0.0, 0.5)

        r2 = np.arange(n_g, 2 * n_g)
        X2[r2, a] = 1.0                    # away offence
        X2[r2, n_t + h] = 1.0              # home defence
        X2[r2, 2 * n_t] = np.where(neutral, 0.0, -0.5)

        y2 = np.concatenate([
            played["home_points"].to_numpy(dtype=float),
            played["away_points"].to_numpy(dtype=float),
        ])

        league_ppg = float(np.mean(y2)) if len(y2) else 27.0
        b02 = np.concatenate([
            np.full(n_t, league_ppg / 2.0) + prior / 4.0,
            np.full(n_t, league_ppg / 2.0) - prior / 4.0,
            [config.HFA_PRIOR],
        ])
        P2 = np.eye(2 * n_t + 1) * (lam * 1.5)
        A2 = X2.T @ X2 + P2
        rhs2 = X2.T @ y2 + P2 @ b02
        beta2 = np.linalg.solve(A2, rhs2)

        played_counts = (
            played["home_team"].value_counts()
            .add(played["away_team"].value_counts(), fill_value=0)
        )

        out = pd.DataFrame({
            "team": teams,
            "rating": rating,
            "off": beta2[:n_t],
            "def": beta2[n_t:2 * n_t],
            "played": [float(played_counts.get(t, 0)) for t in teams],
        })
        out.attrs["hfa"] = hfa
        out.attrs["league_ppg"] = league_ppg
        return out

    # -- public API ---------------------------------------------------------
    def ratings_before(self, year: int, week: int) -> pd.DataFrame:
        """Ratings using only games completed before `week` of `year`."""
        key = (year, week)
        if key in self._cache:
            return self._cache[key]

        g = self.games
        mask = (
            (g["season"] == year)
            & (g["week"] < week)
            & g["home_points"].notna()
            & g["away_points"].notna()
        )
        played = g.loc[mask]
        result = self._solve(played, year)
        self._cache[key] = result
        return result

    def final_ratings(self, year: int) -> pd.Series:
        """End-of-season ratings, used only as the *next* year's prior."""
        if year in self._final_cache:
            return self._final_cache[year]
        g = self.games
        mask = (g["season"] == year) & g["home_points"].notna() & g["away_points"].notna()
        res = self._solve(g.loc[mask], year)
        series = res.set_index("team")["rating"] if not res.empty else pd.Series(dtype=float)
        self._final_cache[year] = series
        return series

    def lookup(self, year: int, week: int, team: str) -> dict:
        table = self.ratings_before(year, week)
        if table.empty:
            return {"rating": 0.0, "off": 13.5, "def": 13.5, "played": 0.0,
                    "hfa": config.HFA_PRIOR, "known": 0}
        row = table.loc[table["team"] == team]
        hfa = table.attrs.get("hfa", config.HFA_PRIOR)
        if row.empty:
            prior = self.priors.by_year.get(year, pd.Series(dtype=float))
            fallback = float(prior.get(team, prior.get(FCS, -20.0))) if len(prior) else -20.0
            half = table.attrs.get("league_ppg", 27.0) / 2.0
            return {"rating": fallback, "off": half, "def": half,
                    "played": 0.0, "hfa": hfa, "known": 0}
        r = row.iloc[0]
        return {"rating": float(r["rating"]), "off": float(r["off"]),
                "def": float(r["def"]), "played": float(r["played"]),
                "hfa": float(hfa), "known": 1}
