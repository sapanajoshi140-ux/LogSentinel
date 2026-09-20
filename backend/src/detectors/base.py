# Detector ABC: fit() / score() / predict()
"""
Common interface every detector (EWMA, Markov, BiLSTM) implements, so DAWF
can fuse them without knowing anything about their internals.

Design note: score() returns a continuous anomaly score, not a hard label.
DAWF fuses scores, and Platt scaling calibrates them into probabilities
downstream (src/calibration/platt.py) — mixing that in at the detector level
would couple concerns that need to vary independently during the ablation
and calibration studies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Detector(ABC):
    """One anomaly detector operating on parsed log sequences.

    A "sequence" is whatever src/splitting produced: a list of template IDs
    for one HDFS block, or one BGL sliding window. Detectors don't care which
    — that's exactly why EWMA/Markov/BiLSTM can share one fusion mechanism.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "Detector":
        """Fit on training sequences. labels is None for unsupervised
        detectors (EWMA, Markov) and required for supervised ones (BiLSTM)."""
        raise NotImplementedError

    @abstractmethod
    def score(self, sequences: list[list[int]]) -> np.ndarray:
        """Return a continuous anomaly score per sequence. Higher = more
        anomalous. Scale need not match across detectors — DAWF fuses
        decisions/weighted votes, not raw scores directly; see fusion/dawf.py."""
        raise NotImplementedError

    def predict(self, sequences: list[list[int]], threshold: float) -> np.ndarray:
        """Convenience: binary predictions from score() at a given threshold."""
        return (self.score(sequences) >= threshold).astype(int)

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{self.name}: save() not yet implemented")

    def load(self, path: str) -> "Detector":
        raise NotImplementedError(f"{self.name}: load() not yet implemented")

    def __repr__(self) -> str:
        return f"<Detector:{self.name}>"
