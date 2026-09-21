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

    # -- metadata helpers ---------------------------------------------------
    # These operate on a RAW sequence (list of template IDs), not on the
    # transformed count vector, and are separate from fit()/transform() on
    # purpose: normalize=True (proportions) is exactly what EWMA needs to
    # avoid the length-bias bug, but proportions also HIDE how long or how
    # novel a sequence actually is -- two sequences with identical
    # proportions can have wildly different absolute volume or novelty.
    # These helpers keep that information available separately, for
    # features, diagnostics, or explainability that specifically needs it
    # (e.g. "this sequence's anomaly involves an unusually long run of
    # never-seen-before events", which proportions alone can't say).
    #
    # "Unknown" here means the same thing it means to transform(): a token
    # not present in the vocabulary learned by fit() -- this covers both
    # UNSEEN_TEMPLATE_ID (a template the PARSER never saw) and, defensively,
    # any other template ID that was simply never in the training data.

    def _require_fitted(self) -> None:
        if not self._index and not self.vocab_:
            raise RuntimeError("call fit() before using this method")

    def is_unknown(self, template_id: int) -> bool:
        """Whether a single template ID falls outside the fitted vocabulary."""
        self._require_fitted()
        return template_id not in self._index

    def sequence_length(self, sequence: list[int]) -> int:
        """Raw token count. Kept separate from the (possibly normalized)
        count vector so absolute volume is never silently lost."""
        return len(sequence)

    def unknown_template_count(self, sequence: list[int]) -> int:
        """How many tokens in this sequence are unknown (may count the
        same unknown template ID more than once, if it repeats)."""
        self._require_fitted()
        return sum(1 for t in sequence if self.is_unknown(t))

    def unknown_template_ratio(self, sequence: list[int]) -> float:
        """unknown_template_count / sequence_length. 0.0 for an empty
        sequence (nothing to be unknown), not a division error."""
        if len(sequence) == 0:
            return 0.0
        return self.unknown_template_count(sequence) / len(sequence)

    def n_distinct_unknown_templates(self, sequence: list[int]) -> int:
        """Number of DISTINCT unknown template IDs, as opposed to
        unknown_template_count's raw occurrence count. A sequence with one
        unknown template repeated 20 times and a sequence with 20
        different unknown templates both have unknown_template_count=20,
        but very different values here -- the first is "one novel event,
        happened a lot", the second is "many different novel events"."""
        self._require_fitted()
        return len({t for t in sequence if self.is_unknown(t)})

    def longest_consecutive_unknown_run(self, sequence: list[int]) -> int:
        """Longest streak of back-to-back unknown tokens. A sequence with
        occasional isolated unknown tokens scattered through otherwise
        normal activity is a different situation from one with a long
        unbroken run of unknowns (e.g. the system entered some state that
        generates only never-before-seen log lines) -- this distinguishes
        the two even when unknown_template_count is identical."""
        self._require_fitted()
        longest = current = 0
        for t in sequence:
            if self.is_unknown(t):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest