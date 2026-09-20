"""
Markov-chain detector.

WHAT IT DOES, IN PLAIN TERMS
    EWMA counts how often each event happens overall, but ignores order.
    Markov does the opposite: it learns "after event A, what usually comes
    next", from training data. On new data, if it sees a jump that almost
    never happens in training (event A followed by event Z, when A is
    normally followed by B or C), it flags that as unusual -- even if event
    Z by itself is common elsewhere. This is exactly what EWMA can't see.

HOW IT WORKS
    1. fit(): build a table of "given the last `order` events, what came
       next, and how often" from training sequences only. order=1 (default)
       means "just look at the previous single event"; higher order looks
       further back.
    2. Every sequence is padded with `order` START markers at the front, so
       even the very first real event in a sequence gets scored against
       "what normally comes first", not skipped.
    3. Laplace (add-one-ish) smoothing means a transition that was simply
       never SEEN in training still gets a small, nonzero probability,
       instead of an impossible zero. That's different from a template
       being UNSEEN_TEMPLATE_ID (a template the PARSER never saw at all) --
       those are handled separately, as an automatic maximum-surprisal
       case, since the parser has already told us that token is novel.
    4. score(): for each sequence, walk through it step by step and measure
       how "surprised" the model is at each step (-log of the probability
       it assigned to what actually happened). AVERAGE the surprisal across
       the sequence, not sum -- summing would make long sequences score
       higher just for being long, the exact same length-bias bug found and
       fixed in EWMA's raw-count features. Averaging keeps the score about
       "how unusual is this sequence's transitions", not "how many
       transitions does it have".
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from src.detectors.base import Detector
from src.features.counts import UNSEEN_TEMPLATE_ID

START_SYMBOL = -2  # padding marker, distinct from UNSEEN_TEMPLATE_ID (-1)


class MarkovDetector(Detector):
    name = "markov"

    def __init__(self, order: int = 1, smoothing: float = 1.0, unseen_penalty: float = 20.0):
        """
        order: how many previous events form the "context" used to predict
            the next one. 1 = "just the previous event" (a classic Markov
            chain). Higher catches longer patterns but needs more training
            data to fill in the (bigger) table of contexts.
        smoothing: Laplace smoothing strength. Higher = more willing to
            treat a never-seen-in-training transition as merely "unlikely"
            rather than "basically impossible".
        unseen_penalty: fixed surprisal (in the same -log-probability units
            as everything else) assigned whenever UNSEEN_TEMPLATE_ID shows
            up in a sequence's context or next-event. This is NOT looked up
            in the table -- the parser already told us this template was
            never seen at all during training, so there is nothing
            meaningful to look up. 20.0 is deliberately far above what a
            merely-rare-but-known transition would score, so a genuinely
            novel template always dominates the sequence's score.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0")

        self.order = order
        self.smoothing = smoothing
        self.unseen_penalty = unseen_penalty

        self.vocab_: list[int] = []
        self._vocab_size = 0
        self._transition_counts: dict[tuple, Counter] = defaultdict(Counter)
        self._context_totals: dict[tuple, int] = defaultdict(int)
        self._fitted = False

    # -- internals ----------------------------------------------------------

    def _padded(self, sequence: list[int]) -> list[int]:
        return [START_SYMBOL] * self.order + list(sequence)

    def _transitions(self, sequence: list[int]):
        """Yield (context_tuple, next_token) pairs for one padded sequence."""
        padded = self._padded(sequence)
        for i in range(self.order, len(padded)):
            context = tuple(padded[i - self.order:i])
            yield context, padded[i]

    def _log_prob(self, context: tuple, next_token: int) -> float:
        """Laplace-smoothed log P(next_token | context), over the REAL
        vocabulary only. Contexts never seen in training fall back
        naturally to a uniform smoothed estimate -- no special-casing
        needed, since count=0 and total=0 plug into the same formula."""
        count = self._transition_counts[context][next_token]
        total = self._context_totals[context]
        prob = (count + self.smoothing) / (total + self.smoothing * self._vocab_size)
        return math.log(prob)

    # -- public API -----------------------------------------------------------

    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "MarkovDetector":
        if not sequences:
            raise ValueError("fit() requires at least one training sequence")

        vocab = sorted({
            t for seq in sequences for t in seq if t != UNSEEN_TEMPLATE_ID
        })
        self.vocab_ = vocab
        self._vocab_size = max(len(vocab), 1)  # avoid div-by-zero if train data is degenerate

        for seq in sequences:
            clean_seq = [t for t in seq if t != UNSEEN_TEMPLATE_ID]
            for context, next_token in self._transitions(clean_seq):
                self._transition_counts[context][next_token] += 1
                self._context_totals[context] += 1

        self._fitted = True
        return self

    def score(self, sequences: list[list[int]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        if not sequences:
            return np.array([])

        scores = np.zeros(len(sequences))
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                scores[i] = 0.0
                continue

            surprisals = []
            for context, next_token in self._transitions(seq):
                if next_token == UNSEEN_TEMPLATE_ID or UNSEEN_TEMPLATE_ID in context:
                    surprisals.append(self.unseen_penalty)
                else:
                    surprisals.append(-self._log_prob(context, next_token))

            # Mean, not sum -- see module docstring: summing would make
            # longer sequences score higher just for having more
            # transitions, independent of whether anything is actually
            # unusual. Same length-bias fix already applied to EWMA.
            scores[i] = float(np.mean(surprisals)) if surprisals else 0.0

        return scores

    def most_surprising_transition(self, sequence: list[int]):
        """For one sequence, return (prev_template, next_template,
        surprisal) for whichever step was the most unexpected. prev_template
        is None if that step was the very first event (context is all
        START_SYMBOL). This is what src/explain/ will call to turn a bare
        score into "this fired because event X unexpectedly followed
        event Y"."""
        if not self._fitted:
            raise RuntimeError("call fit() before most_surprising_transition()")
        if not sequence:
            return None, None, 0.0

        best = (None, None, -1.0)
        for context, next_token in self._transitions(sequence):
            if next_token == UNSEEN_TEMPLATE_ID or UNSEEN_TEMPLATE_ID in context:
                surprisal = self.unseen_penalty
            else:
                surprisal = -self._log_prob(context, next_token)

            if surprisal > best[2]:
                prev = None if context[-1] == START_SYMBOL else context[-1]
                best = (prev, next_token, surprisal)

        return best
