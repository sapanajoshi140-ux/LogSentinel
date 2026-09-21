# runs all four test types, writes results/
"""
Mid-progress evaluation pipeline for EWMA and Markov.

THE METHODOLOGY, IN ORDER (why each step happens where it does)
    1. Fit EWMA and Markov on TRAIN sequences only.
    2. Score VAL sequences with both (frozen) detectors.
    3. Select each detector's own decision threshold on VAL, maximizing
       F1 -- independently per detector, since EWMA's z-scores and
       Markov's surprisal values are on different scales and have no
       shared meaningful threshold (see src/fusion/dawf.py's docstring
       for why raw-score fusion is NOT used for this checkpoint).
    4. FREEZE both detectors and both thresholds. Nothing below this line
       is allowed to look at labels again except to compute a final,
       reported number.
    5. Score TEST sequences with the frozen detectors, apply the frozen
       thresholds, and report EWMA-only, Markov-only, OR-voting (flag if
       EITHER detector says anomaly), and AND-voting (flag only if BOTH
       agree) -- four independent methods, not a blended score.

    "Unknown"-labeled rows (blocks with no entry in anomaly_label.csv) are
    excluded from every precision/recall/F1/confusion-matrix computation,
    at every stage (threshold selection on val AND final reporting on
    test) -- see src/evaluation/metrics.filter_unknown().

Usage:
    python -m src.pipeline.evaluate --data-dir data/processed/hdfs_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.detectors.ewma import EWMADetector
from src.detectors.markov import MarkovDetector
from src.evaluation.metrics import (
    EvaluationResult,
    evaluate_predictions,
    filter_unknown,
    select_threshold,
)
from src.features.loading import load_split_csv_with_raw_labels


def run_evaluation(
    data_dir: Path,
    ewma_alpha: float = 0.3,
    markov_order: int = 1,
) -> dict[str, EvaluationResult]:
    """Runs the full methodology described in the module docstring.
    Returns a dict keyed by method name: "ewma", "markov", "or", "and"."""

    train_seqs, _, _, _ = load_split_csv_with_raw_labels(data_dir / "train.csv")
    val_seqs, _, _, val_raw = load_split_csv_with_raw_labels(data_dir / "val.csv")
    test_seqs, _, _, test_raw = load_split_csv_with_raw_labels(data_dir / "test.csv")

    # -- step 1: fit on train only -------------------------------------------
    ewma = EWMADetector(alpha=ewma_alpha).fit(train_seqs)
    markov = MarkovDetector(order=markov_order).fit(train_seqs)

    # -- step 2 & 3: score val, select thresholds ----------------------------
    # Threshold selection must exclude "Unknown" val rows too -- an
    # unconfirmed row should not influence where the threshold is set,
    # any more than it should influence the final reported metrics.
    ewma_val_scores = ewma.score(val_seqs)
    markov_val_scores = markov.score(val_seqs)

    val_raw_f, ewma_val_scores_f, markov_val_scores_f = filter_unknown(
        val_raw, ewma_val_scores, markov_val_scores
    )
    val_labels_f = np.array([1 if lbl == "Anomaly" else 0 for lbl in val_raw_f])
    ewma_val_scores_f = np.asarray(ewma_val_scores_f)
    markov_val_scores_f = np.asarray(markov_val_scores_f)

    ewma_threshold = select_threshold(ewma_val_scores_f, val_labels_f, metric="f1")
    markov_threshold = select_threshold(markov_val_scores_f, val_labels_f, metric="f1")

    # -- step 4: FROZEN from here on -----------------------------------------
    # (ewma, markov, ewma_threshold, markov_threshold are not modified again)

    # -- step 5: score test, apply frozen thresholds, report ----------------
    ewma_test_scores = ewma.score(test_seqs)
    markov_test_scores = markov.score(test_seqs)

    test_raw_f, ewma_test_scores_f, markov_test_scores_f = filter_unknown(
        test_raw, ewma_test_scores, markov_test_scores
    )
    test_labels_f = np.array([1 if lbl == "Anomaly" else 0 for lbl in test_raw_f])
    ewma_test_scores_f = np.asarray(ewma_test_scores_f)
    markov_test_scores_f = np.asarray(markov_test_scores_f)

    ewma_preds = (ewma_test_scores_f >= ewma_threshold).astype(int)
    markov_preds = (markov_test_scores_f >= markov_threshold).astype(int)
    or_preds = ((ewma_preds == 1) | (markov_preds == 1)).astype(int)
    and_preds = ((ewma_preds == 1) & (markov_preds == 1)).astype(int)

    results = {
        "ewma": evaluate_predictions(ewma_preds, test_labels_f, scores=ewma_test_scores_f),
        "markov": evaluate_predictions(markov_preds, test_labels_f, scores=markov_test_scores_f),
        # OR/AND are combinations of already-thresholded decisions, not a
        # single continuous score of their own -- PR-AUC is not applicable
        # (evaluate_predictions leaves it as None when scores is omitted).
        "or": evaluate_predictions(or_preds, test_labels_f),
        "and": evaluate_predictions(and_preds, test_labels_f),
    }

    return {
        "results": results,
        "thresholds": {"ewma": ewma_threshold, "markov": markov_threshold},
        "n_test_excluded_unknown": len(test_raw) - len(test_raw_f),
        "n_val_excluded_unknown": len(val_raw) - len(val_raw_f),
    }


def print_report(report: dict) -> None:
    print(f"\nThresholds selected on validation split (frozen before touching test):")
    for name, t in report["thresholds"].items():
        print(f"  {name}: {t:.4f}")

    if report["n_val_excluded_unknown"] or report["n_test_excluded_unknown"]:
        print(f"\nExcluded as 'Unknown': {report['n_val_excluded_unknown']} val rows, "
              f"{report['n_test_excluded_unknown']} test rows")

    print("\n--- Results on test split (frozen detectors + thresholds) ---")
    header = f"{'method':10} {'precision':>10} {'recall':>10} {'f1':>8} {'pr_auc':>8} {'tp':>7} {'fp':>7} {'fn':>7} {'tn':>9} {'pred_anom':>10}"
    print(header)
    for name, res in report["results"].items():
        d = res.as_dict()
        auc_str = f"{d['pr_auc']:.4f}" if d["pr_auc"] is not None else "n/a"
        print(f"{name:10} {d['precision']:10.4f} {d['recall']:10.4f} {d['f1']:8.4f} "
              f"{auc_str:>8} {d['tp']:7d} {d['fp']:7d} {d['fn']:7d} {d['tn']:9d} "
              f"{d['n_predicted_anomalies']:10d}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EWMA/Markov mid-progress evaluation pipeline")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/hdfs_split"))
    ap.add_argument("--ewma-alpha", type=float, default=0.3)
    ap.add_argument("--markov-order", type=int, default=1)
    ap.add_argument("--output-json", type=Path, default=None,
                     help="Optional path to save results as JSON")
    args = ap.parse_args()

    for name in ("train.csv", "val.csv", "test.csv"):
        if not (args.data_dir / name).exists():
            print(f"ERROR: {args.data_dir / name} does not exist. "
                  f"Run parsing + splitting first (see README.md).")
            raise SystemExit(1)

    report = run_evaluation(args.data_dir, args.ewma_alpha, args.markov_order)
    print_report(report)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "thresholds": report["thresholds"],
            "n_val_excluded_unknown": report["n_val_excluded_unknown"],
            "n_test_excluded_unknown": report["n_test_excluded_unknown"],
            "results": {k: v.as_dict() for k, v in report["results"].items()},
        }
        with args.output_json.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()