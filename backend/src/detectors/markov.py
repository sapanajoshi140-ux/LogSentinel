"""
Markov-chain detector.

Catches sequence/order anomalies — a transition between templates that
almost never happens normally (e.g. a "terminate" event with no preceding
"receive"). Blind to long-range context beyond the chosen order, which is
why BiLSTM exists alongside it.

TODO — implementation plan:
    1. In fit(): build an order-k transition probability table
       P(template_t | template_(t-1), ..., template_(t-k)) from training
       sequences only. config.detectors.markov_order (default 1) sets k.
       Use add-one (Laplace) smoothing so an unseen-but-plausible transition
       doesn't get probability zero — that's a modeling choice, not the same
       thing as leakage_free_parser's UNSEEN_TEMPLATE_ID, which represents a
       genuinely novel *template*, not a novel *transition* between known
       templates.
    2. In score(): for each sequence, compute the negative log-likelihood of
       its transition sequence under the learned table. Low-probability
       transitions drive the score up.
    3. Decide how to handle UNSEEN_TEMPLATE_ID tokens arriving from the
       parser (src/parsing/leakage_free_parser.py) — these should probably
       score as maximally anomalous by construction, since the parser has
       already told you the template itself was never seen in training.
"""

from __future__ import annotations

import numpy as np

from src.detectors.base import Detector


class MarkovDetector(Detector):
    name = "markov"

    def __init__(self, order: int = 1, smoothing: float = 1.0):
        self.order = order
        self.smoothing = smoothing
        self._transition_counts: dict | None = None

    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "MarkovDetector":
        raise NotImplementedError(
            "MarkovDetector.fit: build an order-k transition table with "
            "Laplace smoothing. See module docstring."
        )

    def score(self, sequences: list[list[int]]) -> np.ndarray:
        raise NotImplementedError(
            "MarkovDetector.score: negative log-likelihood of each "
            "sequence's transitions under the learned table."
        )
