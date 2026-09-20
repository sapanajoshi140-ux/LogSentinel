"""
EWMA (Exponentially Weighted Moving Average) detector.

Catches simple statistical shifts in event frequency — e.g. a template that
normally appears once per sequence suddenly appearing five times. Blind to
event ORDER, which is why it's paired with Markov (order-sensitive) and
BiLSTM (long-range context) in DAWF rather than used alone.

HOW IT WORKS
    1. fit(): each training sequence becomes an event-count vector (one
       column per known template, via src/features/counts.py). Walking
       through sequences IN ORDER, maintain a running EWMA mean and
       variance per template:
           mean_t = alpha * x_t + (1 - alpha) * mean_(t-1)
           var_t  = alpha * (x_t - mean_t)**2 + (1 - alpha) * var_(t-1)
       After the last training sequence, mean/var are FROZEN.
    2. score(): each sequence gets a z-score per template — how many
       standard deviations its count is from the learned mean — and the
       detector reports the MAX z-score across templates as that
       sequence's anomaly score. Max rather than sum because it answers
       "what's the single most unusual thing here", which also happens to
       be more useful for explainability later (SHAP can point at that
       one template).

WHY fit() DOESN'T KEEP ADAPTING DURING score()
    Real EWMA control charts often keep updating online forever. This one
    doesn't, on purpose: continuous online adaptation is DAWF's job (via
    Weighted Majority reweighting), not each individual detector's. If
    EWMA silently kept adapting too, drift-adaptation would be happening
    in two uncoordinated places, and the ablation/drift studies in
    src/evaluation/ would no longer cleanly isolate DAWF's contribution.

WHY UNSEEN TEMPLATES DON'T NEED SPECIAL-CASING
    A template absent from training has a learned mean of ~0 and tiny
    variance, so ANY nonzero count for it in test data produces a huge
    z-score automatically through the same formula. No separate penalty
    term needed — it falls out of the vectorizer's dedicated "unknown"
    bucket (src/features/counts.py) plus the z-score math.
"""

from __future__ import annotations

import numpy as np

from src.detectors.base import Detector
from src.features.counts import EventCountVectorizer


class EWMADetector(Detector):
    name = "ewma"

    def __init__(self, alpha: float = 0.3, normalize: bool = True):
        """
        normalize: passed to EventCountVectorizer. Defaults to True after
        discovering on real HDFS_v1 data that raw counts let a block's
        sheer LENGTH masquerade as anomalous frequency -- several genuinely
        Normal blocks scored ~200,000 (vs. a true-anomaly max of ~3,000)
        under raw counts, purely because they were long-lived blocks with
        more log lines, not because anything unusual happened. See
        src/features/counts.py for the full explanation.
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.vectorizer = EventCountVectorizer(normalize=normalize)
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._fitted = False

    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "EWMADetector":
        if not sequences:
            raise ValueError("fit() requires at least one training sequence")

        self.vectorizer.fit(sequences)
        X = self.vectorizer.transform(sequences)

        mean = np.zeros(X.shape[1])
        var = np.ones(X.shape[1])  # start at 1, not 0, so early z-scores aren't div-by-zero

        for x in X:
            mean = self.alpha * x + (1 - self.alpha) * mean
            var = self.alpha * (x - mean) ** 2 + (1 - self.alpha) * var

        self._mean, self._var = mean, var
        self._fitted = True
        return self

    def score(self, sequences: list[list[int]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        if not sequences:
            return np.array([])

        X = self.vectorizer.transform(sequences)
        std = np.sqrt(np.maximum(self._var, 1e-6))
        z = np.abs(X - self._mean) / std
        return z.max(axis=1)

    def most_deviant_template(self, sequence: list[int]) -> tuple[int | None, float]:
        """For one sequence, return (template_id, z_score) of whichever
        template drove the anomaly score. template_id is None if it's the
        trailing "unknown" bucket that fired. This is what src/explain/
        will call to turn a bare score into "this fired because of
        template X"."""
        X = self.vectorizer.transform([sequence])[0]
        std = np.sqrt(np.maximum(self._var, 1e-6))
        z = np.abs(X - self._mean) / std
        idx = int(np.argmax(z))
        template_id = self.vectorizer.vocab_[idx] if idx < len(self.vectorizer.vocab_) else None
        return template_id, float(z[idx])
