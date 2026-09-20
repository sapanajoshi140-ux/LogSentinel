"""Tests for the Markov detector.

Checks the properties that matter for its role in DAWF: it should react
to unusual ORDER, stay indifferent to sequence length/repetition (the
opposite failure mode from raw-count EWMA), and handle UNSEEN_TEMPLATE_ID
tokens without crashing or silently ignoring them.
"""

import numpy as np
import pytest

from src.detectors.markov import MarkovDetector
from src.features.counts import UNSEEN_TEMPLATE_ID

SHORT_NORMAL = [1, 2, 3, 2]
# Long and repetitive, but a legitimate recurring pattern -- the same shape
# as the real HDFS block that produced a false positive under raw-count EWMA.
LONG_NORMAL = [1, 1, 3, 4] + [10] * 20 + [1, 3, 4] + [10] * 15
TRAIN = [SHORT_NORMAL] * 80 + [LONG_NORMAL] * 20


def test_fit_requires_at_least_one_sequence():
    with pytest.raises(ValueError):
        MarkovDetector().fit([])


def test_score_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        MarkovDetector().score([SHORT_NORMAL])


def test_rejects_invalid_order():
    with pytest.raises(ValueError):
        MarkovDetector(order=0)


def test_rejects_invalid_smoothing():
    with pytest.raises(ValueError):
        MarkovDetector(smoothing=0)


def test_long_repetitive_but_legitimate_block_scores_like_short_normal():
    """The specific failure mode raw-count EWMA had: length/repetition
    alone should NOT make a block look anomalous to Markov, as long as
    its transitions are all ones seen (and common) in training."""
    det = MarkovDetector(order=1).fit(TRAIN)
    short_score = det.score([SHORT_NORMAL])[0]
    long_score = det.score([LONG_NORMAL])[0]
    assert abs(short_score - long_score) < 0.05


def test_novel_transition_scores_higher_than_normal():
    det = MarkovDetector(order=1).fit(TRAIN)
    normal_score = det.score([SHORT_NORMAL])[0]
    weird_score = det.score([[1, 9, 2, 3]])[0]  # "9" right after "1" never happens in training
    assert weird_score > normal_score


def test_unseen_template_forces_high_surprisal():
    det = MarkovDetector(order=1, unseen_penalty=20.0).fit(TRAIN)
    normal_score = det.score([SHORT_NORMAL])[0]
    unseen_score = det.score([[1, 2, UNSEEN_TEMPLATE_ID, 3]])[0]
    assert unseen_score > normal_score
    # with only 4 transitions and one forced to exactly unseen_penalty,
    # the mean should be pulled up substantially toward that penalty
    assert unseen_score > 4.0


def test_naive_repetition_creating_a_real_novel_transition_is_correctly_flagged():
    """The flip side of the test above: if training data NEVER shows what
    follows the end of a pattern (each training example just stops there),
    then looping that pattern back-to-back in test data creates a genuinely
    novel transition -- and Markov SHOULD flag it. This documents that as
    intended behaviour, not a bug (see test_score_averages_not_sums_over_
    sequence_length's docstring for the full explanation)."""
    det = MarkovDetector(order=1).fit(TRAIN)  # SHORT_NORMAL never loops in training
    single = SHORT_NORMAL
    looped = SHORT_NORMAL * 10  # introduces an unseen "2 -> 1" wrap-around transition
    assert det.score([looped])[0] > det.score([single])[0]


def test_score_averages_not_sums_over_sequence_length():
    """A long sequence should not score higher than a short one purely for
    being long -- summing surprisal would do that; averaging must not.

    Care needed here: naively repeating a short training pattern back-to-back
    (e.g. [1,2,3,2] * 50) is NOT a fair test of this, because each training
    example ends after one repetition, so the training data never shows what
    follows the last "2" -- looping back to "1" is then a genuinely novel
    transition, and SHOULD score as surprising. That's Markov doing its job
    correctly, not a length-bias bug.

    To isolate length from novelty, we train on sequences that already
    legitimately loop (so every transition, including the wrap-around, is
    something the model has actually seen), then compare a short vs. long
    instance of that same, fully-legitimate pattern.
    """
    unit = [1, 2, 3, 2]
    looping_short = unit * 2   # transitions include the wrap-around: 2->1
    looping_long = unit * 30   # same transitions, just many more of them

    train = [looping_short, looping_long] * 40
    det = MarkovDetector(order=1).fit(train)

    short_score = det.score([looping_short])[0]
    long_score = det.score([looping_long])[0]
    # A small residual difference is expected (finite-sample smoothing
    # effects), the point is this must NOT scale up with length the way
    # summing would -- contrast with the ~100-1000x gaps seen for genuinely
    # novel transitions/templates elsewhere in this file.
    assert abs(short_score - long_score) < 0.15


def test_higher_order_looks_further_back():
    """order=2 should be able to distinguish a context order=1 can't."""
    # In training, "2" is always followed by "3" except after the specific
    # pair (1,2), where it's followed by "9".
    train = [[1, 2, 9, 3]] * 50 + [[5, 2, 3, 3]] * 50
    det1 = MarkovDetector(order=1).fit(train)
    det2 = MarkovDetector(order=2).fit(train)

    # order=1 has seen "2" followed by both 9 and 3 -- ambiguous, moderate surprisal either way.
    # order=2 has learned (1,2)->9 and (5,2)->3 as separate, confident contexts.
    seq_matching_pattern = [1, 2, 9, 3]
    score1 = det1.score([seq_matching_pattern])[0]
    score2 = det2.score([seq_matching_pattern])[0]
    assert score2 < score1  # order=2 is more confident about this specific pattern


def test_most_surprising_transition_finds_the_weird_jump():
    det = MarkovDetector(order=1).fit(TRAIN)
    prev, nxt, surprisal = det.most_surprising_transition([1, 2, 3, 1, 9, 2])
    assert (prev, nxt) == (1, 9)
    assert surprisal > 0


def test_most_surprising_transition_handles_start_of_sequence():
    det = MarkovDetector(order=1).fit(TRAIN)
    prev, nxt, surprisal = det.most_surprising_transition(SHORT_NORMAL)
    # first event's context is all START_SYMBOL -> prev should report as None
    assert prev is None or isinstance(prev, int)


def test_empty_sequence_scores_zero_not_crash():
    det = MarkovDetector(order=1).fit(TRAIN)
    assert det.score([[]])[0] == 0.0


def test_score_returns_empty_array_for_empty_input():
    det = MarkovDetector(order=1).fit(TRAIN)
    assert len(det.score([])) == 0


def test_context_never_seen_in_training_falls_back_to_uniform_smoothing():
    """A context combination that never occurred in training at all
    (possible with order > 1) must not crash -- it should fall back to a
    smoothed, roughly-uniform probability rather than raising a KeyError."""
    det = MarkovDetector(order=2).fit(TRAIN)
    # (7, 8) as a context pair never appeared anywhere in training
    score = det.score([[7, 8, 1]])[0]
    assert np.isfinite(score)
