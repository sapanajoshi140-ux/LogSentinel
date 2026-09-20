"""
BiLSTM detector.

Catches complex, long-range contextual patterns the other two detectors
can't see. Slowest, least transparent, most data-hungry of the three — the
reason it's fused rather than used alone, per Yu et al. (2024), who found
classical methods sometimes match or beat deep models depending on scenario.

TODO — implementation plan:
    1. Embeddings: template IDs -> dense vectors. Use a Word2Vec-style
       architecture (Mikolov et al., 2013) trained on template sequences as
       if they were sentences — see src/features/embeddings.py (not yet
       written). Train embeddings on the TRAINING split only, same leakage
       principle as the parser.
    2. Architecture: bidirectional LSTM over the embedded sequence ->
       predict the next template ID (like DeepLog) OR a binary
       normal/anomaly label directly, depending on whether you want an
       unsupervised next-event-prediction detector (comparable to DeepLog,
       Du et al. 2017) or a supervised classifier. Decide this before
       writing fit() — it changes the loss function and what labels fit()
       requires.
    3. Anomaly score: for next-event-prediction framing, score = how far
       down the predicted-probability ranking the ACTUAL next event fell
       (DeepLog's approach). For direct classification, score = sigmoid
       output.
    4. config.detectors.bilstm_hidden_dim / bilstm_window / bilstm_epochs
       are already wired up in config/*.yaml — read them in __init__.
    5. This is the detector most likely to need GPU time and the most data
       — HDFS's ~575k blocks should be plenty; watch out on BGL, which is
       smaller and imbalanced (~3% anomalous), so consider class weighting
       or oversampling in the loss.
"""

from __future__ import annotations

import numpy as np

from src.detectors.base import Detector


class BiLSTMDetector(Detector):
    name = "bilstm"

    def __init__(self, hidden_dim: int = 64, window: int = 10, epochs: int = 10):
        self.hidden_dim = hidden_dim
        self.window = window
        self.epochs = epochs
        self._model = None  # torch.nn.Module, once built

    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "BiLSTMDetector":
        raise NotImplementedError(
            "BiLSTMDetector.fit: train embeddings on train split, then a "
            "BiLSTM (next-event-prediction or direct classification — "
            "decide which, see module docstring)."
        )

    def score(self, sequences: list[list[int]]) -> np.ndarray:
        raise NotImplementedError(
            "BiLSTMDetector.score: rank-based score (DeepLog-style) or "
            "sigmoid output, matching whatever fit() trained."
        )
