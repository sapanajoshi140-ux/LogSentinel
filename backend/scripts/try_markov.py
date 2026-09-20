"""
Run the Markov detector against REAL split HDFS (or BGL) data.

Usage (from inside backend/, venv active):
    python scripts/try_markov.py
    python scripts/try_markov.py --train data/processed/bgl_split/train.csv --test data/processed/bgl_split/test.csv --order 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.detectors.markov import MarkovDetector
from src.features.loading import load_split_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check Markov on real split data")
    ap.add_argument("--train", default="data/processed/hdfs_split/train.csv")
    ap.add_argument("--test", default="data/processed/hdfs_split/test.csv")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    train_path, test_path = Path(args.train), Path(args.test)
    for p in (train_path, test_path):
        if not p.exists():
            print(f"ERROR: {p} does not exist. Run parsing + splitting first.")
            sys.exit(1)

    train_seqs, train_labels, train_ids = load_split_csv(train_path)
    test_seqs, test_labels, test_ids = load_split_csv(test_path)

    print(f"train: {len(train_seqs)} sequences, {sum(train_labels)} anomalous "
          f"({sum(train_labels) / len(train_labels):.2%})")
    print(f"test:  {len(test_seqs)} sequences, {sum(test_labels)} anomalous "
          f"({sum(test_labels) / len(test_labels):.2%})")

    det = MarkovDetector(order=args.order).fit(train_seqs)
    scores = det.score(test_seqs)
    test_labels_arr = np.array(test_labels)

    normal_scores = scores[test_labels_arr == 0]
    anomaly_scores = scores[test_labels_arr == 1]

    print(f"\nvocabulary size (templates learned from train): {len(det.vocab_)}")
    print(f"\nnormal  scores: mean={normal_scores.mean():.3f}  median={np.median(normal_scores):.3f}  max={normal_scores.max():.3f}")
    if len(anomaly_scores):
        print(f"anomaly scores: mean={anomaly_scores.mean():.3f}  median={np.median(anomaly_scores):.3f}  min={anomaly_scores.min():.3f}")
        sep = anomaly_scores.mean() - normal_scores.mean()
        print(f"\nmean separation (anomaly - normal): {sep:.3f}  "
              f"{'(anomalies score higher, as expected)' if sep > 0 else '(WARNING: anomalies do NOT score higher on average)'}")

    top_idx = np.argsort(scores)[::-1][: args.top_n]
    print(f"\ntop {args.top_n} highest Markov scores in test set:")
    hits = 0
    for i in top_idx:
        marker = "  <-- true anomaly" if test_labels[i] == 1 else ""
        if test_labels[i] == 1:
            hits += 1
        print(f"  {test_ids[i]:25} true_label={test_labels[i]}  score={scores[i]:.3f}{marker}")
    print(f"\n{hits}/{args.top_n} of the top-scoring sequences are true anomalies.")


if __name__ == "__main__":
    main()
