"""Model fitting, calibrated win probability, and walk-forward evaluation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

import config
from features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


class FeatureMismatchError(RuntimeError):
    """Saved model was fitted on a different feature set than the code builds."""


# The two ratings-derived baselines the trees correct rather than replace.
BASELINE_MARGIN = "proj_margin"
BASELINE_TOTAL = "proj_total"


def _regressor(**kw) -> HistGradientBoostingRegressor:
    params = dict(
        loss="absolute_error",   # margins are heavy-tailed; MAE is the honest loss
        max_iter=450,
        learning_rate=0.045,
        max_depth=None,
        max_leaf_nodes=24,
        min_samples_leaf=45,
        l2_regularization=1.2,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=35,
        random_state=config.RANDOM_SEED,
    )
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def _fit_line(x: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept, with a safe fallback.

    The ratings baseline is systematically compressed early in a season, when
    ridge shrinkage pulls every team toward its preseason prior. Rescaling it
    against reality before the trees see it means the model inherits a
    baseline that is on the right scale, not merely the right order.
    """
    ok = np.isfinite(x) & np.isfinite(target)
    if ok.sum() < 200 or np.std(x[ok]) < 1e-6:
        return 1.0, 0.0
    slope, intercept = np.polyfit(x[ok], target[ok], 1)
    # Guard against a degenerate fit inverting or exploding the baseline.
    if not np.isfinite(slope) or not (0.2 <= slope <= 5.0):
        return 1.0, 0.0
    return float(slope), float(intercept)




@dataclass
class CFBModel:
    """Margin, total, and win probability, fitted together."""

    margin: HistGradientBoostingRegressor | None = None
    total: HistGradientBoostingRegressor | None = None
    margin_line: tuple[float, float] = (1.0, 0.0)
    total_line: tuple[float, float] = (1.0, 0.0)
    wp_coef: float = 0.0
    wp_intercept: float = 0.0
    margin_sigma: float = 16.0
    total_sigma: float = 13.0
    # Residual spread as a function of how many games the teams have played.
    # A week-2 forecast leaning on preseason priors deserves less confidence
    # than a week-12 one, and a single global sigma cannot express that.
    sigma_by_played: list = field(default_factory=list)
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_seasons: list[int] = field(default_factory=list)

    # -- baselines ----------------------------------------------------------
    def _baseline(self, X: pd.DataFrame, which: str) -> np.ndarray:
        col, (a, b) = ((BASELINE_MARGIN, self.margin_line) if which == "margin"
                       else (BASELINE_TOTAL, self.total_line))
        raw = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        return a * np.nan_to_num(raw, nan=0.0) + b

    # -- fitting ------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "CFBModel":
        """Fit the trees on the *residual* from a rescaled ratings baseline.

        Asking a gradient booster to predict a scoring margin outright means
        every prediction is an average of training leaves, so it can never
        exceed what it has already seen and regresses hard toward the middle.
        That is why a 45-point mismatch used to come out as a 20-point
        favourite. Here the ratings difference supplies the level - it is
        linear, so it extrapolates without limit - and the trees only learn
        the correction on top of it.
        """
        X = X[self.features]

        self.margin_line = _fit_line(
            X[BASELINE_MARGIN].to_numpy(dtype=float), y["margin"].to_numpy(dtype=float))
        self.total_line = _fit_line(
            X[BASELINE_TOTAL].to_numpy(dtype=float), y["total"].to_numpy(dtype=float))
        log.info("baseline scaling: margin %.2fx%+.2f, total %.2fx%+.2f",
                 *self.margin_line, *self.total_line)

        margin_base = self._baseline(X, "margin")
        total_base = self._baseline(X, "total")

        self.margin = _regressor().fit(X, y["margin"].to_numpy() - margin_base)
        self.total = _regressor().fit(X, y["total"].to_numpy() - total_base)

        resid = y["margin"].to_numpy() - (margin_base + self.margin.predict(X))
        # 1.4826 * MAD is a robust sd estimate, less swayed by 60-point games.
        self.margin_sigma = float(max(9.0, 1.4826 * np.median(np.abs(resid - np.median(resid)))))
        tresid = y["total"].to_numpy() - (total_base + self.total.predict(X))
        self.total_sigma = float(max(7.0, 1.4826 * np.median(np.abs(tresid - np.median(tresid)))))

        self.sigma_by_played = self._fit_sigma_curve(X, resid)
        self.trained_seasons = sorted(y["season"].unique().tolist())
        self._fit_win_probability(X, y)
        return self

    # -- uncertainty as a function of evidence -------------------------------
    def _fit_sigma_curve(self, X: pd.DataFrame, resid) -> list:
        """Measure residual spread separately for thin and thick evidence.

        Early in a season the ratings are mostly preseason prior, so the same
        predicted margin carries far more uncertainty than it does in November.
        Reporting one global sigma makes week-2 forecasts overconfident, which
        is exactly where the model is weakest and where a reader most needs to
        be told so.
        """
        if "min_played" not in X.columns:
            return []
        played = pd.to_numeric(X["min_played"], errors="coerce").to_numpy()
        r = np.asarray(resid, dtype=float)

        curve = []
        for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 99)):
            sel = (played >= lo) & (played < hi) & np.isfinite(r)
            if sel.sum() < 200:
                continue
            block = r[sel]
            sd = float(1.4826 * np.median(np.abs(block - np.median(block))))
            curve.append({"lo": float(lo), "hi": float(hi),
                          "n": int(sel.sum()), "sigma": max(sd, 6.0)})

        if len(curve) < 2:
            return []

        # More evidence cannot make a forecast less certain, so impose that
        # rather than letting bucket noise invert it. Without this the fitted
        # curve can come back slightly tighter for thin evidence, and the
        # adjustment would then make week-2 predictions *more* confident -
        # the exact opposite of the point.
        for i in range(len(curve) - 2, -1, -1):
            curve[i]["sigma"] = max(curve[i]["sigma"], curve[i + 1]["sigma"])

        log.info("residual spread by games played: %s",
                 ", ".join(f"{c['lo']:.0f}-{c['hi']:.0f}: {c['sigma']:.1f}"
                           f" (n={c['n']:,})" for c in curve))
        return curve

    def sigma_for(self, min_played) -> np.ndarray:
        """Per-game residual spread, falling back to the global figure."""
        m = np.asarray(min_played, dtype=float)
        out = np.full(m.shape, self.margin_sigma, dtype=float)
        for c in (self.sigma_by_played or []):
            out = np.where((m >= c["lo"]) & (m < c["hi"]), c["sigma"], out)
        return np.nan_to_num(out, nan=self.margin_sigma)

    def _fit_win_probability(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Map predicted margin to win probability with a one-variable logistic.

        Isotonic regression on blended probabilities was the wrong tool: fitted
        on a few hundred held-out games it produces flat plateaus, so a coin
        flip came out at 61% and a near-certainty got dragged down to 90%. A
        logistic in the predicted margin has two parameters, is monotonic by
        construction, and keeps rising sensibly past the largest mismatch in
        the training data - which is exactly what a 45-point favourite needs.
        """
        wmask = y["home_win"].notna().to_numpy()
        if wmask.sum() < 400:
            # Fall back to a normal CDF over the margin.
            self.wp_coef, self.wp_intercept = 1.0 / self.margin_sigma, 0.0
            return

        seasons = sorted(y["season"].dropna().unique())
        margins, wins = None, None

        if len(seasons) >= 4:
            # Fit the mapping on genuinely out-of-sample margins so it reflects
            # how confident the model deserves to be on games it has not seen.
            cutoff = seasons[-2]
            train = (y["season"] < cutoff).to_numpy()
            held = (y["season"] >= cutoff).to_numpy() & wmask
            if train.sum() >= 800 and held.sum() >= 300:
                shadow_line = _fit_line(
                    X.loc[train, BASELINE_MARGIN].to_numpy(dtype=float),
                    y.loc[train, "margin"].to_numpy(dtype=float))
                base_tr = (shadow_line[0]
                           * np.nan_to_num(X.loc[train, BASELINE_MARGIN].to_numpy(dtype=float))
                           + shadow_line[1])
                shadow = _regressor().fit(X.loc[train],
                                          y.loc[train, "margin"].to_numpy() - base_tr)
                base_ho = (shadow_line[0]
                           * np.nan_to_num(X.loc[held, BASELINE_MARGIN].to_numpy(dtype=float))
                           + shadow_line[1])
                margins = base_ho + shadow.predict(X.loc[held])
                wins = y.loc[held, "home_win"].to_numpy()

        if margins is None:
            margins = (self._baseline(X, "margin") + self.margin.predict(X))[wmask]
            wins = y.loc[wmask, "home_win"].to_numpy()

        clf = LogisticRegression(C=1e6, solver="lbfgs")
        clf.fit(margins.reshape(-1, 1), wins.astype(int))
        self.wp_coef = float(clf.coef_[0][0])
        self.wp_intercept = float(clf.intercept_[0])
        log.info("win probability: p = sigmoid(%.4f * margin %+.4f) "
                 "-> 1 pt of margin is worth %.1f%% at the coin flip",
                 self.wp_coef, self.wp_intercept, 25 * self.wp_coef)

    # -- prediction ---------------------------------------------------------
    def _win_prob(self, margin: np.ndarray,
                  min_played: np.ndarray | None = None) -> np.ndarray:
        """Win probability, widened when the ratings behind it are thin.

        The slope is scaled by how much wider the residuals are for this level
        of evidence, so the same predicted margin yields a less confident
        number in week 2 than in week 12.
        """
        if min_played is None or not self.sigma_by_played:
            scale = 1.0
        else:
            # Capped at 1: this may only ever widen a probability toward a coin
            # flip, never sharpen one. A bucket whose residuals happen to come
            # back tighter than average is far more likely to be noise than a
            # licence for extra confidence, and overconfidence on thin evidence
            # is the costlier mistake.
            scale = np.minimum(1.0, self.margin_sigma / self.sigma_for(min_played))

        if self.wp_coef <= 0:
            sigma = (self.margin_sigma if min_played is None
                     else self.sigma_for(min_played))
            return norm.cdf(margin / sigma)
        z = np.clip(self.wp_coef * margin * scale + self.wp_intercept, -12, 12)
        return 1.0 / (1.0 + np.exp(-z))

    def check_compatible(self, X: pd.DataFrame) -> None:
        """Fail loudly if the saved model predates the current feature set.

        A pickled model carries the exact column list it was fitted on. If the
        feature code has since changed, pandas raises a bare KeyError deep
        inside a library, which tells you nothing about what to do. The answer
        is always the same - retrain - so say that.
        """
        missing = [c for c in self.features if c not in X.columns]
        if not missing:
            return
        raise FeatureMismatchError(
            f"The saved model expects features that this code no longer "
            f"produces: {missing}. The model file is out of date with the "
            f"feature pipeline.\n\n"
            f"Fix: re-run the Bootstrap workflow to retrain. "
            f"(Model was fitted on seasons "
            f"{min(self.trained_seasons, default='?')}-"
            f"{max(self.trained_seasons, default='?')}.)"
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.check_compatible(X)
        Xf = X[self.features]
        margin = self._baseline(Xf, "margin") + self.margin.predict(Xf)
        total = self._baseline(Xf, "total") + self.total.predict(Xf)
        # A total below a floor is nonsense and would poison the score split.
        total = np.clip(total, 17.0, 120.0)
        played = (Xf["min_played"].to_numpy(dtype=float)
                  if "min_played" in Xf.columns else None)
        prob = np.clip(self._win_prob(margin, played), 0.005, 0.995)
        return pd.DataFrame({
            "pred_margin": margin,
            "pred_total": total,
            "home_win_prob": prob,
            "margin_sigma": self.sigma_for(played) if played is not None
                            else np.full(len(Xf), self.margin_sigma),
            "pred_home_points": (total + margin) / 2,
            "pred_away_points": (total - margin) / 2,
        }, index=X.index)


# -- evaluation -------------------------------------------------------------
def walk_forward(feat: pd.DataFrame, min_train_seasons: int = 4) -> pd.DataFrame:
    """Refit at each season boundary and predict the next season only.

    This is the only evaluation that reflects how the model will actually be
    used. A random train/test split would let the model see future games from
    the same season and would flatter it badly.
    """
    from features import training_matrix

    X_all, y_all = training_matrix(feat)
    seasons = sorted(y_all["season"].unique())
    out = []

    for i, season in enumerate(seasons):
        if i < min_train_seasons:
            continue
        train = y_all["season"] < season
        test = y_all["season"] == season
        if train.sum() < 500 or test.sum() == 0:
            continue

        model = CFBModel().fit(X_all.loc[train.values], y_all.loc[train])
        preds = model.predict(X_all.loc[test.values])
        block = y_all.loc[test].join(preds)
        # Carry the comparables through: the comps calibration is fitted on
        # exactly these out-of-sample rows.
        carry = [c for c in ("comp_home_cover_rate", "comp_over_rate", "comp_n")
                 if c in X_all.columns]
        if carry:
            block = block.join(X_all.loc[test.values, carry])
        block["season_tested"] = season
        out.append(block)
        log.info("walk-forward %s: trained on %d, tested on %d",
                 season, int(train.sum()), int(test.sum()))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(oos: pd.DataFrame) -> dict:
    """Metrics that matter, including the ones that are unflattering."""
    if oos.empty:
        return {}

    m = {}
    m["n_games"] = int(len(oos))
    m["margin_mae"] = float(mean_absolute_error(oos["margin"], oos["pred_margin"]))
    m["total_mae"] = float(mean_absolute_error(oos["total"], oos["pred_total"]))

    wp = oos.dropna(subset=["home_win"])
    if len(wp) > 50:
        m["win_logloss"] = float(log_loss(wp["home_win"], wp["home_win_prob"]))
        m["win_accuracy"] = float(
            ((wp["home_win_prob"] > 0.5).astype(int) == wp["home_win"]).mean())
        m["win_brier"] = float(np.mean((wp["home_win_prob"] - wp["home_win"]) ** 2))

        # Expected calibration error: how far predicted probability sits from
        # observed frequency, weighted by how many games land in each bin.
        bins = pd.cut(wp["home_win_prob"], np.linspace(0, 1, 11))
        grouped = wp.groupby(bins, observed=True).agg(
            pred=("home_win_prob", "mean"),
            obs=("home_win", "mean"),
            n=("home_win", "size"))
        m["win_ece"] = float(
            (grouped["pred"] - grouped["obs"]).abs()
            .mul(grouped["n"]).sum() / grouped["n"].sum())
        m["calibration_table"] = [
            {"bin": str(idx), "predicted": round(float(r["pred"]), 4),
             "observed": round(float(r["obs"]), 4), "n": int(r["n"])}
            for idx, r in grouped.iterrows()
        ]

    # --- the real benchmark: the closing line ---
    lined = oos.dropna(subset=["spread", "margin"]).copy()
    if len(lined) > 100:
        # CFBD spreads are quoted from the home team's perspective and are
        # negative when the home team is favoured, so the implied home margin
        # is the negation.
        lined["market_margin"] = -lined["spread"]
        m["market_margin_mae"] = float(
            mean_absolute_error(lined["margin"], lined["market_margin"]))
        m["model_margin_mae_on_lined"] = float(
            mean_absolute_error(lined["margin"], lined["pred_margin"]))
        m["mae_vs_market"] = m["model_margin_mae_on_lined"] - m["market_margin_mae"]

        lined["edge"] = lined["pred_margin"] - lined["market_margin"]
        lined["ats_correct"] = np.where(
            lined["edge"] > 0,
            lined["margin"] > lined["market_margin"],
            lined["margin"] < lined["market_margin"],
        )
        push = np.isclose(lined["margin"], lined["market_margin"])
        graded = lined.loc[~push]
        m["ats_n"] = int(len(graded))
        m["ats_win_pct"] = float(graded["ats_correct"].mean()) if len(graded) else None

        tiers = {}
        for threshold, label in config.EDGE_TIERS:
            sel = graded.loc[graded["edge"].abs() >= threshold]
            if len(sel) >= 30:
                tiers[label] = {
                    "threshold": threshold,
                    "n": int(len(sel)),
                    "ats_win_pct": float(sel["ats_correct"].mean()),
                }
        m["ats_by_tier"] = tiers
        m["breakeven_at_minus_110"] = 0.5238

    totals = oos.dropna(subset=["over_under", "total"]).copy()
    if len(totals) > 100:
        m["market_total_mae"] = float(
            mean_absolute_error(totals["total"], totals["over_under"]))
        m["model_total_mae_on_lined"] = float(
            mean_absolute_error(totals["total"], totals["pred_total"]))

    by_season = (oos.groupby("season_tested")
                    .apply(lambda d: mean_absolute_error(d["margin"], d["pred_margin"]),
                           include_groups=False)
                    .round(3).to_dict())
    m["margin_mae_by_season"] = {int(k): float(v) for k, v in by_season.items()}
    return m


def summarize(metrics: dict) -> str:
    if not metrics:
        return "No metrics (empty evaluation set)."
    lines = [
        f"Games evaluated out-of-sample : {metrics['n_games']:,}",
        f"Margin MAE                    : {metrics['margin_mae']:.2f} pts",
        f"Total MAE                     : {metrics['total_mae']:.2f} pts",
    ]
    if "win_accuracy" in metrics:
        lines += [
            f"Win-pick accuracy             : {metrics['win_accuracy']*100:.1f}%",
            f"Win-prob log loss             : {metrics['win_logloss']:.4f}",
            f"Brier score                   : {metrics['win_brier']:.4f}",
        ]
    if "market_margin_mae" in metrics:
        delta = metrics["mae_vs_market"]
        verdict = "better than" if delta < 0 else "worse than"
        lines += [
            "",
            f"Closing-line margin MAE       : {metrics['market_margin_mae']:.2f} pts",
            f"Model margin MAE (same games) : {metrics['model_margin_mae_on_lined']:.2f} pts",
            f"  -> model is {abs(delta):.2f} pts {verdict} the market",
            f"ATS record (all picks)        : {metrics['ats_win_pct']*100:.1f}% "
            f"on {metrics['ats_n']:,} games (break-even 52.4%)",
        ]
        for label, t in (metrics.get("ats_by_tier") or {}).items():
            lines.append(f"  {label:<8} (edge >= {t['threshold']:.1f}) : "
                         f"{t['ats_win_pct']*100:.1f}% on {t['n']:,}")
    return "\n".join(lines)


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
