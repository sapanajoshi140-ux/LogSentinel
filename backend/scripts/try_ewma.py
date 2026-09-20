"""
Run the EWMA detector against REAL split HDFS (or BGL) data and check
whether it actually separates anomalies from normals -- not just on the
synthetic examples used in tests/test_ewma_detector.py.

Usage (run from inside backend/, with the venv active):
    python scripts/try_ewma.py
    python scripts/try_ewma.py --train data/processed/bgl_split/train.csv --test data/processed/bgl_split/test.csv --alpha 0.3
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/try_ewma.py` from the backend/ root
# without needing to `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.detectors.ewma import EWMADetector
from src.features.loading import load_split_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check EWMA on real split data")
    ap.add_argument("--train", default="data/processed/hdfs_split/train.csv")
    ap.add_argument("--test", default="data/processed/hdfs_split/test.csv")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    train_path, test_path = Path(args.train), Path(args.test)
    for p in (train_path, test_path):
        if not p.exists():
            print(f"ERROR: {p} does not exist.")
            print("Run the parsing + splitting pipeline first -- see README.md.")
            sys.exit(1)

    train_seqs, train_labels, train_ids = load_split_csv(train_path)
    test_seqs, test_labels, test_ids = load_split_csv(test_path)

    if not train_seqs:
        print(f"ERROR: {train_path} loaded but contains zero sequences.")
        sys.exit(1)
    if not test_seqs:
        print(f"ERROR: {test_path} loaded but contains zero sequences.")
        sys.exit(1)

    print(f"train: {len(train_seqs)} sequences, {sum(train_labels)} anomalous "
          f"({sum(train_labels) / len(train_labels):.2%})")
    print(f"test:  {len(test_seqs)} sequences, {sum(test_labels)} anomalous "
          f"({sum(test_labels) / len(test_labels):.2%})")

    det = EWMADetector(alpha=args.alpha).fit(train_seqs)
    scores = det.score(test_seqs)
    test_labels_arr = np.array(test_labels)

    normal_scores = scores[test_labels_arr == 0]
    anomaly_scores = scores[test_labels_arr == 1]

    print(f"\nvocabulary size (templates learned from train): {len(det.vectorizer.vocab_)}")

    print(f"\nnormal  scores: mean={normal_scores.mean():.3f}  "
          f"median={np.median(normal_scores):.3f}  max={normal_scores.max():.3f}")
    if len(anomaly_scores):
        print(f"anomaly scores: mean={anomaly_scores.mean():.3f}  "
              f"median={np.median(anomaly_scores):.3f}  min={anomaly_scores.min():.3f}")
        separation = anomaly_scores.mean() - normal_scores.mean()
        print(f"\nmean separation (anomaly - normal): {separation:.3f}  "
              f"{'(anomalies score higher, as expected)' if separation > 0 else '(WARNING: anomalies do NOT score higher on average)'}")
    else:
        print("\nWARNING: zero anomalous sequences in this test split -- "
              "can't check separation. Try without --sample, or check your labels file.")

    top_idx = np.argsort(scores)[::-1][: args.top_n]
    print(f"\ntop {args.top_n} highest EWMA scores in test set:")
    hits = 0
    for i in top_idx:
        marker = "  <-- true anomaly" if test_labels[i] == 1 else ""
        if test_labels[i] == 1:
            hits += 1
        print(f"  {test_ids[i]:25} true_label={test_labels[i]}  score={scores[i]:.3f}{marker}")
    print(f"\n{hits}/{args.top_n} of the top-scoring sequences are true anomalies "
          f"(EWMA alone -- expect this to be imperfect, it's blind to event order; "
          f"that's Markov's job).")


if __name__ == "__main__":
    main()
