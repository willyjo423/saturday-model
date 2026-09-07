"""Model fitting, calibrated win probability, and walk-forward evaluation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, mean_absolute_error

import config
from features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


class FeatureMismatchError(RuntimeError):
    """Saved model was fitted on a different feature set than the code builds."""


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


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.045,
        max_leaf_nodes=20,
        min_samples_leaf=60,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=config.RANDOM_SEED,
    )


@dataclass
class CFBModel:
    """Margin, total, and win probability, fitted together."""

    margin: HistGradientBoostingRegressor | None = None
    total: HistGradientBoostingRegressor | None = None
    winner: HistGradientBoostingClassifier | None = None
    calibrator: IsotonicRegression | None = None
    margin_sigma: float = 16.0
    total_sigma: float = 13.0
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_seasons: list[int] = field(default_factory=list)

    # -- fitting ------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "CFBModel":
        X = X[self.features]
        self.margin = _regressor().fit(X, y["margin"])
        self.total = _regressor().fit(X, y["total"])

        resid = y["margin"] - self.margin.predict(X)
        # 1.4826 * MAD is a robust sd estimate, less swayed by 60-point games.
        self.margin_sigma = float(max(9.0, 1.4826 * np.median(np.abs(resid - np.median(resid)))))
        tresid = y["total"] - self.total.predict(X)
        self.total_sigma = float(max(7.0, 1.4826 * np.median(np.abs(tresid - np.median(tresid)))))

        wmask = y["home_win"].notna()
        if wmask.sum() > 200:
            self.winner = _classifier().fit(X.loc[wmask.values], y.loc[wmask, "home_win"])
            self._fit_calibrator(X, y)

        self.trained_seasons = sorted(y["season"].unique().tolist())
        return self

    def _fit_calibrator(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit isotonic calibration on genuinely held-out predictions.

        Calibrating on the training tail is the classic trap: the boosted
        models are near-perfect in sample, so the isotonic map learns to trust
        confidence levels that don't survive contact with new games. Instead we
        hold out the most recent season, fit a shadow model on everything
        before it, and calibrate on the shadow's out-of-sample probabilities.
        """
        seasons = sorted(y["season"].dropna().unique())
        if len(seasons) < 3:
            return

        # More holdout seasons means a less lumpy isotonic fit, but the shadow
        # model needs enough history to be representative. Two when we can
        # afford it, one otherwise.
        n_holdout = 2 if len(seasons) >= 5 else 1
        cutoff = seasons[-n_holdout]
        train = (y["season"] < cutoff).to_numpy()
        held = (y["season"] >= cutoff).to_numpy()
        if train.sum() < 800 or held.sum() < 250:
            return

        shadow = CFBModel(features=list(self.features))
        shadow.margin = _regressor().fit(X.loc[train], y.loc[train, "margin"])
        wtrain = train & y["home_win"].notna().to_numpy()
        shadow.winner = _classifier().fit(X.loc[wtrain], y.loc[wtrain, "home_win"])
        sresid = y.loc[train, "margin"] - shadow.margin.predict(X.loc[train])
        shadow.margin_sigma = float(max(
            9.0, 1.4826 * np.median(np.abs(sresid - np.median(sresid)))))

        eval_mask = held & y["home_win"].notna().to_numpy()
        if eval_mask.sum() < 250:
            return

        raw = shadow._raw_win_prob(X.loc[eval_mask])
        truth = y.loc[eval_mask, "home_win"].to_numpy()
        self.calibrator = IsotonicRegression(
            y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(raw, truth)
        log.info("calibrated win probabilities on %d held-out games (%s+)",
                 int(eval_mask.sum()), int(cutoff))

    # -- prediction ---------------------------------------------------------
    def _raw_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Blend the classifier with a normal CDF over the margin prediction.

        The classifier picks up patterns the margin model smooths over; the
        CDF keeps probabilities coherent with the predicted spread. Averaging
        them is more stable than either alone.
        """
        margin_pred = self.margin.predict(X[self.features])
        cdf_prob = norm.cdf(margin_pred / self.margin_sigma)
        if self.winner is None:
            return cdf_prob
        clf_prob = self.winner.predict_proba(X[self.features])[:, 1]
        return 0.5 * cdf_prob + 0.5 * clf_prob

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
        margin = self.margin.predict(Xf)
        total = self.total.predict(Xf)
        raw = self._raw_win_prob(X)
        prob = self.calibrator.predict(raw) if self.calibrator is not None else raw
        prob = np.clip(prob, 0.01, 0.99)
        return pd.DataFrame({
            "pred_margin": margin,
            "pred_total": total,
            "home_win_prob": prob,
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
