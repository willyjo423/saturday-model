"""Verify the forecast tracker against a synthetic archive of known quality.

    python test_track.py

The tracker's whole job is to turn scattered misses into evidence, so it has
to be right about the arithmetic. Here it grades an archive whose errors were
planted deliberately - a known bias, a known accuracy, a known blowup - and
the checks confirm it recovers them.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import track

results: list[tuple[bool, str]] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    results.append((bool(cond), label))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


RNG = np.random.default_rng(7)

# Ground truth we plant and expect the tracker to recover.
TRUE_BIAS = 3.0          # forecasts over-rate the home team by 3
TRUE_NOISE = 12.0


def build_archive(root: Path, n_files: int = 4, per_file: int = 40) -> pd.DataFrame:
    """Write archive files, and return the actual results to grade against."""
    archive = root / "archive"
    archive.mkdir(parents=True)

    truth, gid = [], 5000
    for f in range(n_files):
        games = []
        for _ in range(per_file):
            gid += 1
            actual = float(RNG.normal(3, 17))
            pred = actual + TRUE_BIAS + float(RNG.normal(0, TRUE_NOISE))
            total_actual = float(max(20, RNG.normal(52, 12)))
            played = int(RNG.integers(0, 10))
            games.append({
                "game_id": gid,
                "week": 1 + f * 3,
                "home_team": f"Home {gid}", "away_team": f"Away {gid}",
                "neutral_site": False,
                "games_played": played,
                "forecast": {
                    "margin": round(pred, 1),
                    "total": round(total_actual + float(RNG.normal(0, 9)), 1),
                    "home_win_prob": round(
                        float(1 / (1 + np.exp(-pred / 16))), 4),
                },
                "market_spread": round(-(actual + float(RNG.normal(0, 11))) * 2) / 2,
                "market_total": 52.0,
            })
            truth.append({"game_id": gid, "actual_margin": actual,
                          "actual_total": total_actual,
                          "home_points": 0, "away_points": 0})

        (archive / f"2026-09-{10 + f:02d}.json").write_text(json.dumps(
            {"season": 2026, "dates": [f"2026-09-{10 + f:02d}"], "games": games}))

    return pd.DataFrame(truth)


def main() -> int:
    print("Grading a synthetic archive with planted errors\n" + "-" * 52)
    root = Path(tempfile.mkdtemp())
    try:
        truth = build_archive(root)

        forecasts = track.load_forecasts(root / "archive")
        check(len(forecasts) == 160, "every archived forecast loaded",
              f"{len(forecasts)} rows")

        # A game forecast twice must be graded once, on the later view.
        dupes = root / "archive" / "2026-09-20.json"
        first = json.loads((root / "archive" / "2026-09-10.json").read_text())
        repeat = dict(first)
        repeat["games"] = first["games"][:5]
        for g in repeat["games"]:
            g["forecast"] = {**g["forecast"], "margin": 99.0}
        dupes.write_text(json.dumps(repeat))

        again = track.load_forecasts(root / "archive")
        check(len(again) == 160, "a re-forecast game is not double counted",
              f"{len(again)} rows")
        refreshed = again.loc[again["game_id"] == first["games"][0]["game_id"],
                              "pred_margin"].iloc[0]
        check(refreshed == 99.0, "the latest forecast supersedes earlier ones")

        graded = track.grade(again, truth)
        check(len(graded) == 160, "all forecasts matched to results",
              f"{len(graded)} graded")

        summary = track.summarise(graded)

        # The planted bias is +3; the five tampered rows drag it up a little.
        check(2.0 < summary["margin_bias"] < 6.0,
              "recovers the planted home-side bias",
              f"{summary['margin_bias']:+.2f} (planted {TRUE_BIAS:+.1f})")
        check(9.0 < summary["margin_mae"] < 16.0,
              "recovers the planted error scale",
              f"MAE {summary['margin_mae']:.2f} (noise sd {TRUE_NOISE})")
        check(0.5 < summary["winner_pct"] < 0.95,
              "winner accuracy plausible", f"{summary['winner_pct']*100:.1f}%")
        check(summary["worst_miss"] >= summary["margin_mae"],
              "worst miss is at least the average miss")

        check("market_mae" in summary and "vs_market" in summary,
              "compares against the market on the same games",
              f"market {summary.get('market_mae')} vs model {summary['margin_mae']}")

        check(bool(summary.get("calibration")),
              "win-probability calibration table produced",
              f"{len(summary.get('calibration', []))} bands")
        cal_ok = all(0.5 <= c["claimed"] <= 1.0 and 0.0 <= c["actual"] <= 1.0
                     for c in summary.get("calibration", []))
        check(cal_ok, "calibration bands are well formed")

        check(bool(summary.get("by_evidence")),
              "breaks results down by how much evidence the ratings had",
              f"{len(summary.get('by_evidence', []))} buckets")
        check(bool(summary.get("by_week")), "breaks results down by season stage")
        check(bool(summary.get("by_role")), "breaks results down by favourite/underdog")

        text = track.report(summary, graded)
        check("FORECAST TRACK RECORD" in text and "Biggest misses" in text,
              "report renders with the misses named")
        check("Graded games" in text and str(summary["n_games"]) in text,
              "report states the sample size")

        # Empty archive must be handled, not crash.
        empty = Path(tempfile.mkdtemp())
        check(track.load_forecasts(empty / "nothing").empty,
              "a missing archive returns empty rather than raising")
        check("No graded forecasts" in track.report({}, pd.DataFrame()),
              "empty report says so plainly")
        shutil.rmtree(empty, ignore_errors=True)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for ok, _ in results if ok)
    print(f"\n{'=' * 52}\n{passed}/{len(results)} checks passed")
    if passed < len(results):
        for ok, label in results:
            if not ok:
                print(f"  - {label}")
    print("=" * 52)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
