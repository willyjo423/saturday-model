"""Grade past forecasts against what actually happened.

    python track.py              # score everything in docs/archive
    python track.py --write      # also write docs/performance.json

Why this exists
---------------
A single bad miss tells you almost nothing. Margins have a spread of roughly
sixteen points, so a forty-point error should turn up on most slates and says
nothing about whether the model is broken. The only way to learn from a game
like that is to stop looking at it alone and start accumulating them.

Every daily run already archives its predictions. This reads that archive,
fetches the real results, and answers the questions one game never can: is the
model worse in September than in November, is it biased toward home teams or
favourites, and are the win probabilities honest - does a set of 65% calls
actually win about 65% of the time?

Nothing here feeds back into the model automatically. It produces evidence; a
change to the model remains a decision a human makes, on purpose, when the
evidence supports it.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from api import CFBDClient, CFBDError, MissingKeyError
from schema import GAME_FIELDS, normalise

log = logging.getLogger(__name__)

ARCHIVE = config.DOCS / "archive"
OUTPUT = config.DOCS / "performance.json"


# ---------------------------------------------------------------- loading
def load_forecasts(archive: Path = ARCHIVE) -> pd.DataFrame:
    """Every archived prediction, one row per game, newest file wins.

    The same game appears in several archives when a run looks days ahead, so
    later forecasts supersede earlier ones - grading a Monday guess about
    Saturday alongside Friday's would double-count and flatter the fresher one.
    """
    if not archive.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(archive.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue

        for g in payload.get("games", []):
            fc = g.get("forecast") or {}
            if g.get("game_id") is None:
                continue
            rows.append({
                "game_id": int(g["game_id"]),
                "forecast_file": path.stem,
                "season": payload.get("season"),
                "week": g.get("week"),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "pred_margin": fc.get("margin"),
                "pred_total": fc.get("total"),
                "home_win_prob": fc.get("home_win_prob"),
                "market_spread": g.get("market_spread"),
                "market_total": g.get("market_total"),
                "games_played": g.get("games_played"),
                "neutral_site": g.get("neutral_site"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("forecast_file")
    return df.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def fetch_results(client: CFBDClient, seasons: list[int]) -> pd.DataFrame:
    """Final scores for the seasons the archive touches."""
    frames = []
    for season in seasons:
        try:
            raw = client.all_games(int(season))
        except CFBDError as exc:
            log.warning("results for %s unavailable: %s", season, exc)
            continue
        if raw.empty:
            continue
        g = normalise(raw, GAME_FIELDS)
        frames.append(g)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ("game_id", "home_points", "away_points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["game_id", "home_points", "away_points"])
    df["actual_margin"] = df["home_points"] - df["away_points"]
    df["actual_total"] = df["home_points"] + df["away_points"]
    return df[["game_id", "home_points", "away_points",
               "actual_margin", "actual_total"]].drop_duplicates("game_id")


# ---------------------------------------------------------------- scoring
def grade(forecasts: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty or results.empty:
        return pd.DataFrame()

    df = forecasts.merge(results, on="game_id", how="inner")
    df = df.dropna(subset=["pred_margin", "actual_margin"])
    if df.empty:
        return df

    df["margin_error"] = df["pred_margin"] - df["actual_margin"]
    df["abs_margin_error"] = df["margin_error"].abs()
    df["home_won"] = (df["actual_margin"] > 0).astype(int)
    df["called_home"] = (df["home_win_prob"] > 0.5).astype(int)
    df["winner_correct"] = (df["called_home"] == df["home_won"]).astype(int)

    has_total = df["pred_total"].notna() & df["actual_total"].notna()
    df.loc[has_total, "total_error"] = (
        df.loc[has_total, "pred_total"] - df.loc[has_total, "actual_total"])
    df["abs_total_error"] = df["total_error"].abs()

    lined = df["market_spread"].notna()
    df.loc[lined, "market_margin"] = -df.loc[lined, "market_spread"]
    df.loc[lined, "market_abs_error"] = (
        df.loc[lined, "market_margin"] - df.loc[lined, "actual_margin"]).abs()
    return df


def _bucket_table(df: pd.DataFrame, by: pd.Series, label: str) -> list[dict]:
    out = []
    for name, block in df.groupby(by, observed=True):
        if len(block) < 5:
            continue
        row = {
            label: str(name),
            "n": int(len(block)),
            "margin_mae": round(float(block["abs_margin_error"].mean()), 2),
            "winner_pct": round(float(block["winner_correct"].mean()), 3),
            "bias": round(float(block["margin_error"].mean()), 2),
        }
        lined = block.dropna(subset=["market_abs_error"])
        if len(lined) >= 5:
            row["market_mae"] = round(float(lined["market_abs_error"].mean()), 2)
            row["vs_market"] = round(
                float(lined["abs_margin_error"].mean()
                      - lined["market_abs_error"].mean()), 2)
        out.append(row)
    return out


def summarise(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    out = {
        "n_games": int(len(df)),
        "margin_mae": round(float(df["abs_margin_error"].mean()), 2),
        # A positive bias means the home team was systematically over-rated.
        "margin_bias": round(float(df["margin_error"].mean()), 2),
        "winner_pct": round(float(df["winner_correct"].mean()), 3),
        "worst_miss": round(float(df["abs_margin_error"].max()), 1),
    }

    tot = df.dropna(subset=["abs_total_error"])
    if len(tot):
        out["total_mae"] = round(float(tot["abs_total_error"].mean()), 2)

    lined = df.dropna(subset=["market_abs_error"])
    if len(lined) >= 10:
        out["market_mae"] = round(float(lined["market_abs_error"].mean()), 2)
        out["vs_market"] = round(
            float(lined["abs_margin_error"].mean()
                  - lined["market_abs_error"].mean()), 2)

    # Are the probabilities honest? A set of 65% calls should win about 65%.
    wp = df.dropna(subset=["home_win_prob"]).copy()
    if len(wp) >= 20:
        wp["conf"] = np.where(wp["home_win_prob"] >= 0.5,
                              wp["home_win_prob"], 1 - wp["home_win_prob"])
        bins = pd.cut(wp["conf"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      include_lowest=True)
        cal = []
        for name, block in wp.groupby(bins, observed=True):
            if len(block) < 5:
                continue
            cal.append({
                "band": str(name),
                "n": int(len(block)),
                "claimed": round(float(block["conf"].mean()), 3),
                "actual": round(float(block["winner_correct"].mean()), 3),
            })
        out["calibration"] = cal

    # The breakdowns a single game can never give you.
    if df["games_played"].notna().any():
        played = pd.cut(pd.to_numeric(df["games_played"], errors="coerce"),
                        [-1, 1, 3, 6, 99],
                        labels=["0-1 games", "2-3 games", "4-6 games", "7+ games"])
        out["by_evidence"] = _bucket_table(df, played, "evidence")

    if df["week"].notna().any():
        wk = pd.cut(pd.to_numeric(df["week"], errors="coerce"),
                    [-1, 3, 7, 11, 99],
                    labels=["weeks 1-3", "weeks 4-7", "weeks 8-11", "week 12+"])
        out["by_week"] = _bucket_table(df, wk, "period")

    lined_side = df.dropna(subset=["market_spread"]).copy()
    if len(lined_side) >= 20:
        role = np.where(lined_side["market_spread"] < 0,
                        "home favourite", "home underdog")
        out["by_role"] = _bucket_table(lined_side, pd.Series(role, index=lined_side.index),
                                       "role")
    return out


def report(summary: dict, df: pd.DataFrame) -> str:
    if not summary:
        return ("No graded forecasts yet. The archive fills up as the daily "
                "job runs and results come in.")

    L = ["=" * 62, "FORECAST TRACK RECORD", "=" * 62,
         f"Graded games        : {summary['n_games']:,}",
         f"Margin MAE          : {summary['margin_mae']:.2f} pts",
         f"Home-side bias      : {summary['margin_bias']:+.2f} pts "
         f"({'over' if summary['margin_bias'] > 0 else 'under'}-rating the home team)",
         f"Winners called      : {summary['winner_pct'] * 100:.1f}%",
         f"Worst single miss   : {summary['worst_miss']:.0f} pts"]

    if "total_mae" in summary:
        L.append(f"Total MAE           : {summary['total_mae']:.2f} pts")
    if "market_mae" in summary:
        L += [f"Closing-line MAE    : {summary['market_mae']:.2f} pts",
              f"  -> model is {abs(summary['vs_market']):.2f} pts "
              f"{'worse' if summary['vs_market'] > 0 else 'better'} than the market"]

    if summary.get("calibration"):
        L += ["", "Win-probability honesty (claimed vs actual):"]
        for c in summary["calibration"]:
            L.append(f"  {c['claimed']*100:5.1f}% claimed -> "
                     f"{c['actual']*100:5.1f}% actual   (n={c['n']:,})")

    for key, title in (("by_evidence", "By how much the ratings had to go on"),
                       ("by_week", "By stage of season"),
                       ("by_role", "By role")):
        rows = summary.get(key)
        if not rows:
            continue
        L += ["", title + ":"]
        for r in rows:
            name = next(v for k, v in r.items() if k not in
                        ("n", "margin_mae", "winner_pct", "bias",
                         "market_mae", "vs_market"))
            extra = (f"  vs market {r['vs_market']:+.2f}"
                     if "vs_market" in r else "")
            L.append(f"  {name:<14} n={r['n']:5,}  MAE {r['margin_mae']:5.2f}  "
                     f"winners {r['winner_pct']*100:5.1f}%  "
                     f"bias {r['bias']:+5.2f}{extra}")

    if not df.empty:
        worst = df.nlargest(min(5, len(df)), "abs_margin_error")
        L += ["", "Biggest misses:"]
        for _, r in worst.iterrows():
            L.append(f"  {r['away_team']} at {r['home_team']}: "
                     f"forecast {r['pred_margin']:+.0f}, "
                     f"actual {r['actual_margin']:+.0f} "
                     f"({r['abs_margin_error']:.0f} off)")

    L.append("=" * 62)
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Grade archived forecasts")
    p.add_argument("--archive", default=str(ARCHIVE))
    p.add_argument("--write", action="store_true",
                   help="write docs/performance.json as well as printing")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    forecasts = load_forecasts(Path(args.archive))
    if forecasts.empty:
        print("No archived forecasts found yet.")
        return 0
    log.info("%d archived forecasts across %d files",
             len(forecasts), forecasts["forecast_file"].nunique())

    seasons = sorted({int(s) for s in forecasts["season"].dropna().unique()})
    try:
        results = fetch_results(CFBDClient(), seasons)
    except MissingKeyError as exc:
        print(f"\n{exc}\n")
        return 2

    graded = grade(forecasts, results)
    if graded.empty:
        print(f"{len(forecasts)} forecasts archived, none with final scores yet.")
        return 0

    summary = summarise(graded)
    print(report(summary, graded))

    if args.write:
        summary["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        OUTPUT.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
