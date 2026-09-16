# Platt scaling [10] + ECE (Guo et al. [11])
"""
Platt scaling (Platt, 1999) — fits a 1-D logistic regression mapping a raw
DAWF fused score to a calibrated probability, so "confidence 0.82" means
something an on-call engineer can actually trust as a probability, not just
an arbitrary score. Calibration is checked using the method from Guo et al.
(2017): reliability diagrams and Expected Calibration Error (ECE).

Fit on the VALIDATION split, never train or test — Platt scaling itself can
overfit and leak if fit on the same data used to train the detectors or
report final numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattScaler:
    """Wraps sklearn's LogisticRegression as the standard 1-D Platt scaler."""

    def __init__(self):
        self._lr = LogisticRegression()
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray) -> "PlattScaler":
        raw_scores = np.asarray(raw_scores).reshape(-1, 1)
        self._lr.fit(raw_scores, true_labels)
        self._fitted = True
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() on the validation split before calibrate()")
        raw_scores = np.asarray(raw_scores).reshape(-1, 1)
        return self._lr.predict_proba(raw_scores)[:, 1]


@dataclass
class CalibrationReport:
    ece: float
    bin_edges: list[float]
    bin_confidence: list[float]   # mean predicted probability per bin
    bin_accuracy: list[float]     # observed fraction positive per bin
    bin_counts: list[int]


def expected_calibration_error(
    probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """ECE per Guo et al. (2017): bin predictions by confidence, compare
    mean confidence to observed accuracy in each bin, weight by bin size.

    ECE = sum_b (|bin_b| / N) * |accuracy(bin_b) - confidence(bin_b)|

    Run this BEFORE and AFTER Platt scaling on the same held-out data and
    report both numbers — that comparison is the actual evidence Platt
    scaling helped, not just an assertion that it should.
    """
    probs = np.asarray(probs)
    true_labels = np.asarray(true_labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_conf, bin_acc, bin_n = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        n_in_bin = int(in_bin.sum())

        if n_in_bin == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            bin_n.append(0)
            continue

        conf = float(probs[in_bin].mean())
        acc = float(true_labels[in_bin].mean())
        ece += (n_in_bin / len(probs)) * abs(acc - conf)

        bin_conf.append(conf)
        bin_acc.append(acc)
        bin_n.append(n_in_bin)

    return CalibrationReport(
        ece=ece,
        bin_edges=bin_edges.tolist(),
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
        bin_counts=bin_n,
    )
