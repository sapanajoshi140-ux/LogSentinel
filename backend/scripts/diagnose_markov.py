"""
Why is a sequence-only detector missing anomalies? Measure the ceiling.

A detector that only looks at the ORDER of template IDs (Markov) cannot flag
an anomalous block whose exact event sequence also occurs among the NORMAL
training blocks -- to the detector it is indistinguishable from normal.
This script counts how many test anomalies are like that. That share is a
hard cap on Markov's recall (max recall = 1 - share), whatever the tuning.

Usage (from inside backend/):
    python scripts/diagnose_markov.py
    python scripts/diagnose_markov.py --data-dir data/processed/bgl_split
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.loading import load_split_csv_with_raw_labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Recall-ceiling diagnostic for sequence-only detectors")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/hdfs_split"))
    args = ap.parse_args()

    for name in ("train.csv", "test.csv"):
        if not (args.data_dir / name).exists():
            print(f"ERROR: {args.data_dir / name} does not exist. Run parsing + splitting first.")
            sys.exit(1)

    train_seqs, _, _, train_raw = load_split_csv_with_raw_labels(args.data_dir / "train.csv")
    test_seqs, _, _, test_raw = load_split_csv_with_raw_labels(args.data_dir / "test.csv")

    print(f"train labels: {dict(Counter(train_raw))}")
    print(f"test labels:  {dict(Counter(test_raw))}")

    normal_train = {tuple(s) for s, r in zip(train_seqs, train_raw) if r == "Normal"}
    print(f"\nunique NORMAL train sequences: {len(normal_train)}")

    test_anoms = [tuple(s) for s, r in zip(test_seqs, test_raw) if r == "Anomaly"]
    test_norms = [tuple(s) for s, r in zip(test_seqs, test_raw) if r == "Normal"]
    if not test_anoms:
        print("No labelled anomalies in test split.")
        return

    hidden = sum(1 for s in test_anoms if s in normal_train)
    novel = len(test_anoms) - hidden
    print(f"\ntest anomalies: {len(test_anoms)}")
    print(f"  with a NEW sequence (detectable by order):          {novel}")
    print(f"  with a sequence identical to normal training blocks: {hidden}")
    print(f"\n=> recall ceiling for any sequence-only detector: {novel / len(test_anoms):.1%}")

    if test_norms:
        novel_norm = sum(1 for s in test_norms if s not in normal_train)
        print(f"\n(For reference, {novel_norm}/{len(test_norms)} test NORMAL blocks also have a "
              f"sequence never seen in normal training -- an unavoidable source of false positives.)")

    def _mean_len(seqs):
        return sum(len(s) for s in seqs) / max(len(seqs), 1)

    print(f"\nmean sequence length: normal={_mean_len(test_norms):.1f}  anomaly={_mean_len(test_anoms):.1f}")


if __name__ == "__main__":
    main()