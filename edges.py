"""Edge calibration: how much of a disagreement with the market is real?

The problem
-----------
Sort games by how far the model sits from the closing line and the biggest
disagreements tend to go *against* the model. This is not bad luck, it is
arithmetic. Write the two estimates as

    model  = truth + our error
    market = truth + a much smaller error

then

    edge = model - market ~= our error

Ranking by edge is therefore close to ranking our own predictions by how wrong
they are, and then backing the worst ones hardest. Textbook adverse selection.

It is compounded by *where* large edges come from. They cluster on the games
where our inputs are weakest: a starting quarterback out that the market knows
about and we do not, a team three games into a season with ratings still mostly
preseason prior, a stale line that has not moved in days.

The fix
-------
Measure, on out-of-sample predictions only, what fraction of a claimed edge
actually shows up in the result, as a function of edge size. Then shrink new
edges by that measured relationship, and tier plays by what each bucket
*actually did* rather than by assuming bigger is better.

Nothing here touches the prediction itself. The model stays independent of the
market; this layer only decides how much to trust the difference.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BREAKEVEN = 0.5238          # -110 juice
BUCKET_EDGES = [0.0, 1.5, 3.0, 4.5, 6.0, 9.0, 13.0, 999.0]
MIN_BUCKET = 60             # below this a bucket's hit rate is meaningless


@dataclass
class EdgeCalibration:
    """A fitted map from claimed edge to trustworthy edge."""

    a: float = 1.0           # linear term
    b: float = 0.0           # curvature: negative means large edges saturate
    intercept: float = 0.0
    buckets: list[dict] = field(default_factory=list)
    n_fitted: int = 0
    fitted: bool = False

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, oos: pd.DataFrame) -> "EdgeCalibration":
        """Fit on walk-forward out-of-sample predictions.

        `oos` needs pred_margin, margin and spread. In-sample data would be
        worse than useless here: the whole quantity being measured is how much
        the model's disagreement survives contact with unseen games.
        """
        need = {"pred_margin", "margin", "spread"}
        if oos is None or oos.empty or not need.issubset(oos.columns):
            log.warning("edge calibration: no usable out-of-sample data")
            return cls()

        d = oos.dropna(subset=["pred_margin", "margin", "spread"]).copy()
        if len(d) < 400:
            log.warning("edge calibration: only %d games, leaving edges unshrunk",
                        len(d))
            return cls()

        d["market_margin"] = -d["spread"]
        d["edge"] = d["pred_margin"] - d["market_margin"]
        d["realized"] = d["margin"] - d["market_margin"]

        e = d["edge"].to_numpy(dtype=float)
        r = d["realized"].to_numpy(dtype=float)

        # realized ~ a*edge + b*edge*|edge| + c. The quadratic term lets the
        # relationship bend over, which is exactly the effect being measured.
        X = np.column_stack([e, e * np.abs(e), np.ones_like(e)])
        coef, *_ = np.linalg.lstsq(X, r, rcond=None)
        a, b, c = (float(coef[0]), float(coef[1]), float(coef[2]))

        obj = cls(a=a, b=b, intercept=c, n_fitted=int(len(d)), fitted=True)
        obj.buckets = obj._bucket_stats(d)

        log.info("edge calibration on %d out-of-sample games: "
                 "realized = %.3f*edge %+.5f*edge*|edge| %+.3f",
                 len(d), a, b, c)
        for bk in obj.buckets:
            log.info("  |edge| %4.1f-%-5.1f  n=%5d  ATS %5.1f%% (+/-%.1f)  "
                     "keeps %5.1f%% of the edge",
                     bk["lo"], bk["hi"], bk["n"], bk["ats"] * 100,
                     bk["se"] * 100 * 1.96, bk["realized_fraction"] * 100)
        return obj

    @staticmethod
    def _bucket_stats(d: pd.DataFrame) -> list[dict]:
        out = []
        mag = d["edge"].abs()
        for lo, hi in zip(BUCKET_EDGES, BUCKET_EDGES[1:]):
            sel = d.loc[(mag >= lo) & (mag < hi)]
            # A push is not a loss; grading it as one biases every bucket down.
            sel = sel.loc[~np.isclose(sel["margin"], sel["market_margin"])]
            if len(sel) < MIN_BUCKET:
                continue
            correct = np.where(sel["edge"] > 0,
                               sel["realized"] > 0, sel["realized"] < 0)
            ats = float(np.mean(correct))
            n = int(len(sel))
            signed = sel["realized"] * np.sign(sel["edge"])
            denom = float(sel["edge"].abs().mean())
            out.append({
                "lo": float(lo), "hi": float(min(hi, 99.0)), "n": n,
                "ats": ats,
                "se": float(np.sqrt(max(ats * (1 - ats), 1e-6) / n)),
                "realized_fraction": (float(signed.mean() / denom)
                                      if denom > 1e-6 else 0.0),
                "mean_edge": float(sel["edge"].abs().mean()),
            })
        return out

    # -- applying -----------------------------------------------------------
    def shrink(self, edge: float | np.ndarray) -> np.ndarray:
        """The part of a claimed edge worth believing."""
        e = np.asarray(edge, dtype=float)
        if not self.fitted:
            return e
        out = self.a * e + self.b * e * np.abs(e) + self.intercept
        # Two guards against the fit doing something silly at the extremes:
        # never flip the sign of an edge, and never amplify one.
        out = np.where(np.sign(out) != np.sign(e), 0.0, out)
        return np.sign(e) * np.minimum(np.abs(out), np.abs(e))

    def bucket_for(self, edge: float) -> dict | None:
        mag = abs(float(edge))
        for bk in self.buckets:
            if bk["lo"] <= mag < bk["hi"]:
                return bk
        return None

    def tier(self, edge: float) -> tuple[str | None, dict | None]:
        """Label a play by what its edge bucket historically did.

        Deliberately not by edge size. If the 10-point bucket has lost money
        historically, it does not get promoted for being large - it gets no
        tier at all, and the dashboard says why.
        """
        bk = self.bucket_for(edge)
        if bk is None:
            # Unmeasured bucket: fall back to size, but never call it Strong.
            mag = abs(float(edge))
            return ("Lean" if mag >= 3.0 else ("Slight" if mag >= 2.0 else None)), None

        ats, se = bk["ats"], bk["se"]
        if ats - se > BREAKEVEN:
            return "Strong", bk
        if ats > BREAKEVEN:
            return "Lean", bk
        if ats > 0.50:
            return "Slight", bk
        return None, bk

    # -- persistence --------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "EdgeCalibration":
        try:
            return cls(**json.loads(text))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read edge calibration (%s); using identity", exc)
            return cls()

    def summary(self) -> str:
        if not self.fitted:
            return "Edge calibration: not fitted (edges used as-is)."
        lines = [f"Edge calibration, fitted on {self.n_fitted:,} out-of-sample games:",
                 f"  realized = {self.a:.3f}*edge {self.b:+.5f}*edge*|edge| "
                 f"{self.intercept:+.3f}"]
        if self.buckets:
            lines.append("  |edge|        n     ATS       95% band      keeps")
            for bk in self.buckets:
                band = 1.96 * bk["se"] * 100
                lines.append(
                    f"  {bk['lo']:4.1f}-{bk['hi']:<5.1f} {bk['n']:6d}  "
                    f"{bk['ats']*100:5.1f}%  "
                    f"{bk['ats']*100-band:5.1f}-{bk['ats']*100+band:<5.1f}%  "
                    f"{bk['realized_fraction']*100:5.1f}%")
            lines.append(f"  break-even at -110 is {BREAKEVEN*100:.1f}%")
        for e in (3.0, 6.0, 10.0, 15.0):
            lines.append(f"  a claimed {e:.0f}-pt edge is treated as "
                         f"{float(self.shrink(e)):.1f} pts")
        return "\n".join(lines)


# -- why a big disagreement might be our fault, not the market's -------------
def confidence_flags(row) -> list[str]:
    """Concrete reasons to distrust a large disagreement on this game.

    A big edge is usually a symptom of missing information rather than an
    insight, so it is worth naming what is missing.
    """
    flags = []

    def val(key, default=np.nan):
        v = row.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    played = val("min_played")
    if not np.isnan(played) and played < 3:
        flags.append("Ratings still mostly preseason - few games played")

    eff_games = val("eff_games")
    if np.isnan(val("home_off_ppa")) or np.isnan(val("away_off_ppa")):
        flags.append("No play-level efficiency for one side yet")
    elif not np.isnan(eff_games) and eff_games < 3:
        flags.append("Efficiency based on very few games")

    disp = val("spread_dispersion")
    if not np.isnan(disp):
        if disp >= 3.0:
            flags.append(f"Books disagree by {disp:.1f} pts - market unsettled")
        elif disp == 0.0 and val("n_providers") <= 1:
            flags.append("Only one book quoted - line may be stale")

    move = val("spread_move")
    if not np.isnan(move) and abs(move) >= 3.0:
        flags.append(f"Line moved {abs(move):.1f} pts since opening")

    if np.isnan(val("temp_f")) and not val("is_dome"):
        flags.append("No weather available")

    return flags
