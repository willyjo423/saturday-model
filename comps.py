"""Historical comparables: what happened when games like this one were played.

The idea
--------
Reduce a matchup to a handful of profile axes - how big the rating gap is, how
lopsided the efficiency edges are, how fast both teams play, whether it is
neutral, how rough the weather is - then find the games from the last decade
that sit closest in that space and look at how they actually turned out.

What this is good for, and what it isn't
----------------------------------------
It is *not* obviously a better point predictor than the gradient booster. Trees
already partition the feature space and average outcomes inside each region,
which is an adaptive version of the same idea, and with 55 features "nearest"
starts to lose meaning. So the comps are exposed as features and the model is
left to decide how much to trust them - if they add nothing, the A/B harness
will say so.

What comps give that a booster cannot:

* **A distribution.** Not "Georgia by 17" but "the middle half of comparable
  games landed between +3 and +31, and the favourite covered 54% of the time."
* **Named precedents** you can eyeball, which is the only real way to sanity
  check whether the model is reasoning sensibly about a specific game.
* **Confidence from agreement.** A six-point edge where every comparable game
  broke the same way is a different proposition from one where they scattered.

Leakage
-------
A game may only draw comparables from games played strictly before it. The
pool is ordered by (season, week) and masked per target, and there is a test
that a game can never be its own neighbour.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# The profile axes. Deliberately few: distance in 50 dimensions is meaningless,
# and these are the dimensions that actually separate one matchup from another.
COMP_AXES = [
    "proj_margin",
    "edge_ppa_home", "edge_ppa_away",
    "edge_success_rate_home", "edge_success_rate_away",
    "edge_explosiveness_home", "edge_explosiveness_away",
    "edge_havoc_home", "edge_havoc_away",
    "pace_sum",
    "neutral_site",
    "min_played",
    "wind_mph",
    "away_travel_mi",
]

# Sensible starting weights, used when no learned importances are supplied.
# The rating gap dominates because it should; everything else refines it.
DEFAULT_WEIGHTS = {
    "proj_margin": 3.0,
    "edge_ppa_home": 1.5, "edge_ppa_away": 1.5,
    "edge_success_rate_home": 1.0, "edge_success_rate_away": 1.0,
    "edge_explosiveness_home": 0.7, "edge_explosiveness_away": 0.7,
    "edge_havoc_home": 0.5, "edge_havoc_away": 0.5,
    "pace_sum": 0.8,
    "neutral_site": 0.6,
    "min_played": 0.8,
    "wind_mph": 0.5,
    "away_travel_mi": 0.3,
}

COMP_FEATURES = [
    "comp_margin_median", "comp_margin_mean", "comp_margin_sd",
    "comp_margin_p25", "comp_margin_p75",
    "comp_total_median", "comp_total_sd",
    "comp_home_win_rate", "comp_agreement",
    "comp_n", "comp_distance",
]

CHUNK = 400


def _time_index(df: pd.DataFrame) -> np.ndarray:
    """A single sortable number per game, so 'played before' is one comparison."""
    season = pd.to_numeric(df["season"], errors="coerce").fillna(0).to_numpy()
    week = pd.to_numeric(df["week"], errors="coerce").fillna(0).to_numpy()
    return season * 100.0 + week


class CompsEngine:
    """Nearest historical matchups, with leakage enforced by construction."""

    def __init__(self, history: pd.DataFrame, weights: dict | None = None,
                 axes: list[str] | None = None):
        self.axes = [a for a in (axes or COMP_AXES) if a in history.columns]
        missing = [a for a in (axes or COMP_AXES) if a not in history.columns]
        if missing:
            log.warning("comps: axes unavailable and skipped: %s", missing)

        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        usable = history["margin"].notna() & history["total"].notna()
        self.history = history.loc[usable].reset_index(drop=True)
        self.available = len(self.history) >= 500 and len(self.axes) >= 4
        if not self.available:
            log.warning("comps: only %d usable historical games across %d axes "
                        "- comparables will be blank",
                        len(self.history), len(self.axes))
            return

        raw = self.history[self.axes].to_numpy(dtype=float)
        # Robust standardisation: median and IQR, so a handful of 60-point
        # blowouts don't set the scale for every axis.
        self.center = np.nanmedian(raw, axis=0)
        q75, q25 = np.nanpercentile(raw, [75, 25], axis=0)
        self.scale = np.where((q75 - q25) > 1e-9, (q75 - q25), 1.0)

        self.w = np.array([self.weights.get(a, 1.0) for a in self.axes], dtype=float)
        self.pool = self._encode(self.history)
        self.pool_time = _time_index(self.history)
        self.pool_margin = self.history["margin"].to_numpy(dtype=float)
        self.pool_total = self.history["total"].to_numpy(dtype=float)
        self.pool_sq = (self.pool ** 2).sum(axis=1)

        log.info("comps: pool of %d games across %d axes",
                 len(self.history), len(self.axes))

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        raw = df[self.axes].to_numpy(dtype=float)
        z = (raw - self.center) / self.scale
        # A missing axis contributes nothing to distance rather than blowing it up.
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z * self.w

    # -- core ---------------------------------------------------------------
    def neighbours(self, targets: pd.DataFrame, k: int = 200
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Indices into the pool and their distances, nearest first.

        Only games strictly earlier than the target are eligible, so a game can
        never be its own comparable and never sees the future.
        """
        n = len(targets)
        idx_out = np.full((n, k), -1, dtype=np.int64)
        dist_out = np.full((n, k), np.inf, dtype=float)
        if not self.available or n == 0:
            return idx_out, dist_out

        enc = self._encode(targets)
        t_time = _time_index(targets)
        enc_sq = (enc ** 2).sum(axis=1)

        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            block = enc[start:stop]
            d2 = (enc_sq[start:stop, None] + self.pool_sq[None, :]
                  - 2.0 * block @ self.pool.T)
            np.maximum(d2, 0.0, out=d2)

            # Mask anything not strictly in the past.
            future = self.pool_time[None, :] >= t_time[start:stop, None]
            d2[future] = np.inf

            take = min(k, d2.shape[1])
            part = np.argpartition(d2, take - 1, axis=1)[:, :take]
            part_d = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(part_d, axis=1)
            nearest = np.take_along_axis(part, order, axis=1)
            nearest_d = np.take_along_axis(part_d, order, axis=1)

            valid = np.isfinite(nearest_d)
            nearest = np.where(valid, nearest, -1)
            idx_out[start:stop, :take] = nearest
            dist_out[start:stop, :take] = np.sqrt(np.where(valid, nearest_d, np.inf))

        return idx_out, dist_out

    def summarise(self, targets: pd.DataFrame, k: int = 200) -> pd.DataFrame:
        """One row of comparable-outcome statistics per target game."""
        idx, dist = self.neighbours(targets, k=k)
        n = len(targets)
        cols = {c: np.full(n, np.nan) for c in COMP_FEATURES}
        cols["comp_n"] = np.zeros(n)

        for i in range(n):
            valid = idx[i] >= 0
            count = int(valid.sum())
            cols["comp_n"][i] = count
            if count < 20:
                continue
            take = idx[i][valid]
            margins = self.pool_margin[take]
            totals = self.pool_total[take]
            median = float(np.median(margins))

            cols["comp_margin_median"][i] = median
            cols["comp_margin_mean"][i] = float(np.mean(margins))
            cols["comp_margin_sd"][i] = float(np.std(margins))
            cols["comp_margin_p25"][i] = float(np.percentile(margins, 25))
            cols["comp_margin_p75"][i] = float(np.percentile(margins, 75))
            cols["comp_total_median"][i] = float(np.median(totals))
            cols["comp_total_sd"][i] = float(np.std(totals))
            cols["comp_home_win_rate"][i] = float(np.mean(margins > 0))
            # How consistently the comparables broke the same way as their
            # own median - a direct read on how much to trust them.
            side = np.sign(median) if median != 0 else 1.0
            cols["comp_agreement"][i] = float(np.mean(np.sign(margins) == side))
            cols["comp_distance"][i] = float(np.mean(dist[i][valid]))

        return pd.DataFrame(cols, index=targets.index)

    def examples(self, target_row: pd.Series, k: int = 5) -> list[dict]:
        """The closest few comparables, named, for showing on a card."""
        if not self.available:
            return []
        frame = target_row.to_frame().T
        idx, dist = self.neighbours(frame, k=max(k, 20))
        out = []
        for pos, d in zip(idx[0], dist[0]):
            if pos < 0 or len(out) >= k:
                continue
            g = self.history.iloc[int(pos)]
            out.append({
                "season": int(g["season"]),
                "week": int(g["week"]),
                "home_team": str(g.get("home_team", "")),
                "away_team": str(g.get("away_team", "")),
                "margin": float(g["margin"]),
                "total": float(g["total"]),
                "distance": round(float(d), 3),
            })
        return out


def attach_comps(feat: pd.DataFrame, engine: "CompsEngine",
                 k: int = 200) -> pd.DataFrame:
    """Write comparable-outcome statistics onto a feature frame in place.

    Assigned rather than joined, because `build_features` has already created
    the comp columns as NaN so that the declared feature list is always
    complete even when comparables are unavailable.
    """
    out = feat.copy()
    if not engine.available:
        for col in COMP_FEATURES:
            out[col] = np.nan
        out["comp_n"] = 0.0
        return out

    summary = engine.summarise(feat, k=k)
    for col in COMP_FEATURES:
        out[col] = summary[col].to_numpy()

    found = float(np.mean(summary["comp_n"] >= 20))
    log.info("comps: %.1f%% of games found a usable set of comparables",
             found * 100)
    return out


def learn_axis_weights(X: pd.DataFrame, y_margin: pd.Series,
                       axes: list[str] | None = None,
                       n_samples: int = 1500, seed: int = 0) -> dict:
    """Weight each comp axis by how much it actually moves the scoreboard.

    Fits a quick booster on scoring margin and measures permutation importance
    for the axis columns. This model exists only to rank the axes - it is not
    the production model - but it means the comparable search emphasises a
    rating gap far more than a travel distance, rather than treating them as
    equally defining.

    Returns an empty dict on any failure, which the caller reads as "use the
    hand-set defaults". Weighting is a refinement, not a requirement.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    axes = [a for a in (axes or COMP_AXES) if a in X.columns]
    if not axes or len(X) < 500:
        return {}

    try:
        rng = np.random.default_rng(seed)
        take = min(n_samples, len(X))
        rows = rng.choice(len(X), size=take, replace=False)
        Xs, ys = X.iloc[rows], y_margin.iloc[rows]

        quick = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=150, learning_rate=0.08,
            max_leaf_nodes=20, min_samples_leaf=40,
            early_stopping=True, validation_fraction=0.15,
            random_state=seed).fit(Xs, ys)

        result = permutation_importance(
            quick, Xs, ys, n_repeats=3, random_state=seed,
            scoring="neg_mean_absolute_error")
    except Exception as exc:  # noqa: BLE001 - a nicety, not a need
        log.warning("comps: axis weighting failed (%s); using defaults", exc)
        return {}

    columns = list(X.columns)
    raw = {a: max(0.0, float(result.importances_mean[columns.index(a)]))
           for a in axes}

    total = sum(raw.values())
    if total <= 0:
        return {}
    # Normalise so the mean weight is 1, keeping distances on a familiar scale.
    mean = total / len(raw)
    weights = {a: round(v / mean, 3) for a, v in raw.items()}
    log.info("comps: learned axis weights %s",
             dict(sorted(weights.items(), key=lambda kv: -kv[1])[:5]))
    return weights
