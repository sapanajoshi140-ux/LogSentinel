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

TWO CHANGES THAT FIXED LOW RECALL (see scripts/diagnose_markov.py)
    1. END-OF-SEQUENCE symbol. Many real anomalies (e.g. HDFS blocks that
       are cut short because a replica never finished) contain ONLY
       normal-looking transitions -- they simply stop too early. Without a
       transition for "the sequence ended here", the detector had nothing
       to be surprised by. With use_end_symbol=True every sequence gets a
       final END token, so "A followed by END" is learned from normal
       data and a truncated sequence is scored as unlikely.
    2. Fit on NORMAL sequences only. If anomalous training blocks are
       included, their odd transitions are counted as normal and their own
       surprisal drops. The caller is responsible for passing normal
       sequences only (see src/pipeline/evaluate.py); "Unknown" rows are
       excluded too, since they are not confirmed normal.

    Because of (2), any template ID that never appeared in the normal
    training sequences is treated exactly like UNSEEN_TEMPLATE_ID at score
    time (automatic unseen_penalty), even if the parser knows that ID.

KNOWN LIMIT (not a bug): a sequence-only detector cannot flag an anomaly
    whose exact event sequence also occurs in normal data. Measure how
    many such anomalies exist with scripts/diagnose_markov.py -- that
    number is the recall ceiling for this detector.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from src.detectors.base import Detector
from src.features.counts import UNSEEN_TEMPLATE_ID

START_SYMBOL = -2  # padding marker, distinct from UNSEEN_TEMPLATE_ID (-1)
END_SYMBOL = -3    # end-of-sequence marker; lets truncated sequences be detected


class MarkovDetector(Detector):
    name = "markov"

    def __init__(
        self,
        order: int = 1,
        smoothing: float = 1.0,
        unseen_penalty: float = 20.0,
        use_end_symbol: bool = True,
    ):
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
        use_end_symbol: append an END token to every sequence so that
            "sequence stopped here" is itself a learned transition and
            truncated sequences get a high score. Set False only to
            reproduce the old behaviour.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0")

        self.order = order
        self.smoothing = smoothing
        self.unseen_penalty = unseen_penalty
        self.use_end_symbol = use_end_symbol

        self._reset_state()

    # -- internals ----------------------------------------------------------

    def _padded(self, sequence: list[int]) -> list[int]:
        padded = [START_SYMBOL] * self.order + list(sequence)
        if self.use_end_symbol:
            padded.append(END_SYMBOL)
        return padded

    def _canonical(self, sequence: list[int]) -> list[int]:
        """Map every token the model never saw in (normal) training to
        UNSEEN_TEMPLATE_ID, so it gets the automatic unseen_penalty."""
        return [t if t in self._vocab_set else UNSEEN_TEMPLATE_ID for t in sequence]

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
        needed, since count=0 and total=0 plug into the same formula.

        Uses .get() rather than [] on purpose: self._transition_counts and
        self._context_totals are defaultdicts, and simply READING a
        missing key with [] auto-inserts it (a real bug found by testing:
        merely calling score() on a sequence with a never-seen context was
        silently mutating the fitted state by inserting empty entries).
        .get() performs a read-only lookup with no such side effect."""
        counter = self._transition_counts.get(context)
        count = counter.get(next_token, 0) if counter is not None else 0
        total = self._context_totals.get(context, 0)
        prob = (count + self.smoothing) / (total + self.smoothing * self._vocab_size)
        return math.log(prob)

    # -- public API -----------------------------------------------------------

    def _reset_state(self) -> None:
        """Clear every piece of learned state. Called at the start of
        fit(), so a second fit() call is a clean re-fit, not an
        accumulation on top of whatever the first fit() learned. Before
        this existed, self._transition_counts and self._context_totals
        (created once in __init__ as defaultdicts) silently kept growing
        across repeated fit() calls, while self.vocab_ was correctly
        replaced -- a real, confirmed bug: refitting on a second dataset
        left the first dataset's transitions mixed in."""
        self.vocab_ = []
        self._vocab_set = set()
        self._vocab_size = 0
        self._transition_counts = defaultdict(Counter)
        self._context_totals = defaultdict(int)
        self._fitted = False

    def fit(self, sequences: list[list[int]], labels: list[int] | None = None) -> "MarkovDetector":
        if not sequences:
            raise ValueError("fit() requires at least one training sequence")

        self._reset_state()

        vocab = sorted({
            t for seq in sequences for t in seq if t != UNSEEN_TEMPLATE_ID
        })
        self.vocab_ = vocab
        self._vocab_set = set(vocab)
        # max(..., 1): if every training sequence is degenerate (empty, or
        # entirely UNSEEN_TEMPLATE_ID), vocab is empty -- avoid div-by-zero
        # in _log_prob rather than crashing on genuinely bad training data.
        # +1 for END_SYMBOL, which is a possible "next event" too.
        self._vocab_size = max(len(vocab), 1) + (1 if self.use_end_symbol else 0)

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
            for context, next_token in self._transitions(self._canonical(seq)):
                if next_token == UNSEEN_TEMPLATE_ID or UNSEEN_TEMPLATE_ID in context:
                    surprisals.append(self.unseen_penalty)
                else:
                    surprisals.append(-self._log_prob(context, next_token))

            # Mean, not sum -- see module docstring: summing would make
            # longer sequences score higher just for having more
            # transitions, independent of whether anything is actually
            # unusual. Same length-bias fix already applied to EWMA.
            scores[i] = float(np.mean(surprisals)) if surprisals else 0.0

        # Safety net: every branch above should already produce a finite
        # value (unseen_penalty is a fixed finite constant, _log_prob's
        # denominator is always > 0 because of Laplace smoothing), but
        # guarantee it rather than assume it -- a NaN/inf score would
        # silently corrupt DAWF fusion and threshold selection downstream.
        return np.nan_to_num(scores, nan=0.0, posinf=self.unseen_penalty, neginf=0.0)

    def most_surprising_transition(self, sequence: list[int]):
        """For one sequence, return (prev_template, next_template,
        surprisal) for whichever step was the most unexpected.

        prev_template is None if that step was the very first event
        (context is all START_SYMBOL). next_template is END_SYMBOL when the
        most surprising thing was the sequence STOPPING where it did
        (a truncated sequence). Both are reported using the ORIGINAL
        template IDs from `sequence`, even for IDs the model treats as
        unseen when scoring, so explanations can name the real template.
        This is what src/explain/ will call to turn a bare score into
        "this fired because event X unexpectedly followed event Y"."""
        if not self._fitted:
            raise RuntimeError("call fit() before most_surprising_transition()")
        if not sequence:
            return None, None, 0.0

        original = list(self._transitions(sequence))
        scored = list(self._transitions(self._canonical(sequence)))

        best = (None, None, -1.0)
        for (orig_ctx, orig_next), (context, next_token) in zip(original, scored):
            if next_token == UNSEEN_TEMPLATE_ID or UNSEEN_TEMPLATE_ID in context:
                surprisal = self.unseen_penalty
            else:
                surprisal = -self._log_prob(context, next_token)

            if surprisal > best[2]:
                prev = None if orig_ctx[-1] == START_SYMBOL else orig_ctx[-1]
                best = (prev, orig_next, surprisal)

        return best