"""Train the production models and record honest out-of-sample metrics.

    python train.py
"""
from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

import config
from storage import load_table, save_table
from features import training_matrix
from model import CFBModel, evaluate, save_metrics, summarize, walk_forward

log = logging.getLogger(__name__)

MODEL_PATH = config.MODELS / "cfb_model.joblib"
METRICS_PATH = config.MODELS / "metrics.json"
BACKTEST_PATH = config.DATA / "backtest"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train CFB prediction models")
    p.add_argument("--data", default=str(config.DATA / "training"))
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--min-train-seasons", type=int, default=4)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    feat = load_table(args.data)
    if feat is None:
        print(f"No dataset at {args.data}. Run build.py first.")
        return 2
    log.info("Loaded %d games (%s-%s)", len(feat),
             feat["season"].min(), feat["season"].max())

    metrics = {}
    if not args.skip_backtest:
        log.info("Running walk-forward backtest...")
        oos = walk_forward(feat, min_train_seasons=args.min_train_seasons)
        if not oos.empty:
            save_table(oos, BACKTEST_PATH)
            metrics = evaluate(oos)
            print("\n" + "=" * 62)
            print("OUT-OF-SAMPLE PERFORMANCE (walk-forward by season)")
            print("=" * 62)
            print(summarize(metrics))
            print("=" * 62 + "\n")

    log.info("Fitting production model on all seasons...")
    X, y = training_matrix(feat)
    model = CFBModel().fit(X, y)
    joblib.dump(model, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)

    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved metrics-> {METRICS_PATH}")
    print(f"Margin sigma : {model.margin_sigma:.2f} pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
