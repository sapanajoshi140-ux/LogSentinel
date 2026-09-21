"""
Run EWMA + Markov together through DAWF, on real split data.

WHY THIS SCRIPT EXISTS, IN PLAIN TERMS
    EWMA's raw scores and Markov's raw scores are on completely different
    scales (EWMA can go into the hundreds; Markov is usually under 5). If
    DAWF just averaged those raw numbers, EWMA would dominate every vote
    purely because its numbers are bigger -- not because it's more
    accurate. That's not a fair fusion.

    The fix: calibrate each detector's raw score into a 0-1 PROBABILITY
    first (via Platt scaling, src/calibration/platt.py), using the
    VALIDATION split (which has real labels) to learn that mapping. Once
    both detectors are speaking the same 0-1 language, DAWF's averaging
    and reweighting actually means something.

THE STEPS
    1. Fit EWMA and Markov on the TRAIN split only (as always).
    2. Score the VAL split with both detectors (raw scores).
    3. Fit a separate Platt scaler per detector on those val scores + val
       labels -- this is the ONLY place labels are used for anything
       other than final evaluation.
    4. Score the TEST split with both detectors, run those raw scores
       through the matching Platt scaler -> calibrated probabilities.
    5. Feed the TEST split through DAWF one sequence at a time, in order:
       predict with current weights, then update weights using the now-
       known true label. This mirrors a real deployment, where you'd only
       learn the true label after the fact (e.g. after an incident is
       confirmed).

Usage (from inside backend/, venv active):
    python scripts/try_dawf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.calibration.platt import PlattScaler, expected_calibration_error
from src.detectors.ewma import EWMADetector
from src.detectors.markov import MarkovDetector
from src.features.loading import load_split_csv
from src.fusion.dawf import DAWF


def load_all_splits(base_dir: Path):
    train = load_split_csv(base_dir / "train.csv")
    val = load_split_csv(base_dir / "val.csv")
    test = load_split_csv(base_dir / "test.csv")
    return train, val, test


def main() -> None:
    base_dir = Path("data/processed/hdfs_split")
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (base_dir / name).exists():
            print(f"ERROR: {base_dir / name} does not exist.")
            print("Run the parsing + splitting pipeline first (see README.md), "
                  "and make sure you have a val.csv, not just train/test.")
            sys.exit(1)

    (train_seqs, train_labels, _), (val_seqs, val_labels, _), (test_seqs, test_labels, test_ids) = \
        load_all_splits(base_dir)

    print(f"train: {len(train_seqs)} sequences, {sum(train_labels)} anomalous")
    print(f"val:   {len(val_seqs)} sequences, {sum(val_labels)} anomalous")
    print(f"test:  {len(test_seqs)} sequences, {sum(test_labels)} anomalous")

    # -- 1. fit both detectors on train only ---------------------------------
    print("\nFitting EWMA and Markov on train split...")
    ewma = EWMADetector(alpha=0.3).fit(train_seqs)
    markov = MarkovDetector(order=1).fit(train_seqs)

    # -- 2 & 3. calibrate each detector's raw scores into probabilities,
    #    using VAL only ---------------------------------------------------
    print("Calibrating each detector's scores into probabilities on val split...")
    val_labels_arr = np.array(val_labels)

    ewma_val_raw = ewma.score(val_seqs)
    markov_val_raw = markov.score(val_seqs)

    ewma_scaler = PlattScaler().fit(ewma_val_raw, val_labels_arr)
    markov_scaler = PlattScaler().fit(markov_val_raw, val_labels_arr)

    ece_ewma = expected_calibration_error(ewma_scaler.calibrate(ewma_val_raw), val_labels_arr)
    ece_markov = expected_calibration_error(markov_scaler.calibrate(markov_val_raw), val_labels_arr)
    print(f"  EWMA   calibration error on val (should be small): {ece_ewma.ece:.4f}")
    print(f"  Markov calibration error on val (should be small): {ece_markov.ece:.4f}")

    # -- 4. score test split, calibrate ---------------------------------------
    print("\nScoring test split with both detectors...")
    ewma_test_raw = ewma.score(test_seqs)
    markov_test_raw = markov.score(test_seqs)

    ewma_test_prob = ewma_scaler.calibrate(ewma_test_raw)
    markov_test_prob = markov_scaler.calibrate(markov_test_raw)

    # sanity check: are the two detectors actually seeing different things,
    # or just agreeing with each other all the time (which would make
    # fusion pointless)?
    disagreement = np.abs(ewma_test_prob - markov_test_prob)
    print(f"  mean |EWMA_prob - Markov_prob| on test: {disagreement.mean():.3f} "
          f"(0 = always agree, higher = genuinely different opinions)")

    # -- 5. run DAWF sequentially over the test split --------------------------
    print("\nRunning DAWF fusion over the test split (in order, online)...")
    dawf = DAWF(detector_names=["ewma", "markov"], min_weight=0.01)
    threshold = 0.5

    fused_preds = np.zeros(len(test_seqs), dtype=int)
    fused_probs = np.zeros(len(test_seqs))

    for i in range(len(test_seqs)):
        scores = {"ewma": float(ewma_test_prob[i]), "markov": float(markov_test_prob[i])}
        fused_prob, fused_pred = dawf.predict(scores, threshold)
        fused_probs[i] = fused_prob
        fused_preds[i] = fused_pred
        dawf.update(scores, true_label=test_labels[i], threshold=threshold)

    # -- results ----------------------------------------------------------------
    test_labels_arr = np.array(test_labels)

    def prf(preds, labels):
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    ewma_preds = (ewma_test_prob >= threshold).astype(int)
    markov_preds = (markov_test_prob >= threshold).astype(int)

    print("\n--- Precision / Recall / F1 @ threshold=0.5 ---")
    for name, preds in [("EWMA alone", ewma_preds), ("Markov alone", markov_preds), ("DAWF fusion", fused_preds)]:
        p, r, f1 = prf(preds, test_labels_arr)
        print(f"  {name:14} precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    print(f"\nFinal DAWF weights after processing test split: {dawf.current_weights()}")
    print("(if these differ noticeably from the 50/50 they started at, that's "
          "DAWF having learned one detector was more reliable on this data)")

    # save weight trajectory for later plotting / drift analysis
    out_dir = Path("results/hdfs")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = dawf.weight_trajectory()
    with (out_dir / "dawf_weight_trajectory.csv").open("w") as f:
        f.write("step,ewma_weight,markov_weight\n")
        for state in traj:
            w = dict(zip(state.detector_names, state.weights))
            f.write(f"{state.n_updates},{w['ewma']:.6f},{w['markov']:.6f}\n")
    print(f"\nWeight trajectory saved to {out_dir / 'dawf_weight_trajectory.csv'} "
          f"({len(traj)} steps) -- plot this to see how weights shifted over time.")


if __name__ == "__main__":
    main()