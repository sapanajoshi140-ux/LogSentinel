"""
Turns a sequence of template IDs into a fixed-size count vector, so
detectors that need numeric features (EWMA today; possibly others later)
don't each reimplement vocabulary handling.

Vocabulary is learned from TRAINING sequences only (fit()), same leakage
principle as the parser: a template the vectorizer never saw in training
should not get its own dimension, it should fall into the trailing
"unknown" bucket. This includes genuinely novel templates the parser
already flagged as UNSEEN_TEMPLATE_ID, and (defensively) any template ID
that reaches transform() without having been in the fitted vocabulary.
"""

from __future__ import annotations

import numpy as np

UNSEEN_TEMPLATE_ID = -1  # must match src/parsing/leakage_free_parser.py


class EventCountVectorizer:
    def __init__(self, normalize: bool = False):
        """
        normalize: if True, transform() returns per-sequence PROPORTIONS
            (counts / sequence length) instead of raw counts. Without this,
            a block that is simply longer than average (more replication
            events, bigger file, nothing anomalous) can rack up a raw count
            far above the training mean purely because of its length, not
            because anything unusual happened -- which produces enormous,
            spurious anomaly scores on genuinely normal blocks. Confirmed
            on real HDFS_v1 data: several Normal blocks scored ~200,000
            (vs. a max true-anomaly score of ~3,000) before this was added.
            Proportions are invariant to sequence length, which removes
            that confound.
        """
        self.normalize = normalize
        self.vocab_: list[int] = []
        self._index: dict[int, int] = {}

    def fit(self, sequences: list[list[int]]) -> "EventCountVectorizer":
        vocab = sorted({
            t for seq in sequences for t in seq if t != UNSEEN_TEMPLATE_ID
        })
        self.vocab_ = vocab
        self._index = {t: i for i, t in enumerate(vocab)}
        return self

    @property
    def n_features(self) -> int:
        return len(self.vocab_) + 1  # +1 = trailing "unknown" bucket

    def transform(self, sequences: list[list[int]]) -> np.ndarray:
        if not self._index and not self.vocab_:
            raise RuntimeError("call fit() before transform()")

        unknown_col = self.n_features - 1
        X = np.zeros((len(sequences), self.n_features))
        for i, seq in enumerate(sequences):
            for t in seq:
                j = self._index.get(t, unknown_col)
                X[i, j] += 1
            if self.normalize and len(seq) > 0:
                X[i] /= len(seq)
        return X
