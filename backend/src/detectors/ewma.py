"""
EWMA (Exponentially Weighted Moving Average) detector.

Catches simple statistical shifts in event frequency — e.g. a template that
normally appears once per sequence suddenly appearing five times. Blind to
event ORDER, which is why it's paired with Markov (order-sensitive) and
BiLSTM (long-range context) in DAWF rather than used alone.

TODO — implementation plan:
    1. In fit(): build an event-count vector per training sequence (one
       column per known template ID — reuse src/features/counts.py once
       that's written). Track a running mean and variance per template
       using the standard EWMA update:
           mean_t = alpha * x_t + (1 - alpha) * mean_(t-1)
       (config.detectors.ewma_alpha, default 0.3 — see config/*.yaml)
    2. In score(): for a new sequence's count vector, compute how many
       standard deviations each template's count is from its EWMA mean.
       Aggregate (e.g. max or sum) across templates into one anomaly score
       per sequence.
    3. threshold in predict() should come from config.detectors.ewma_threshold.

Keep this detector STATEFUL and cheap to update online — its whole value in
DAWF is being the fast, lightweight vote. If it starts requiring full
retraining to update, it stops being distinct from BiLSTM's design point.
"""

from __future__ import annotations

import numpy as np

from src.detectors.base import Detector


class EWMADetector(Detector):
    name = "ewma"

    def __init__(self, alpha: float = 0.3, n_templates: int | None = None):
        self.alpha = alpha
        self.n_templates = n_templates
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None

    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "EWMADetector":
        raise NotImplementedError(
            "EWMADetector.fit: build event-count vectors and initialize "
            "running EWMA mean/variance per template. See module docstring."
        )

    def score(self, sequences: list[list[int]]) -> np.ndarray:
        raise NotImplementedError(
            "EWMADetector.score: compute per-template deviation from the "
            "learned EWMA mean, aggregate into one score per sequence."
        )
