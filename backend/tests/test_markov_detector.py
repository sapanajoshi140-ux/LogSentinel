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
    novel_score = det.score([[1, 9, 2, 3]])[0]
    # Both legitimate blocks must sit far below a block with a genuinely
    # novel transition. (An absolute tolerance is too brittle now that the
    # END transition adds one extra term to every mean; the property that
    # matters is "repetition alone does not look anomalous".)
    assert short_score < 0.15 * novel_score
    assert long_score < 0.15 * novel_score


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
    # The long sequence has 30/2 = 15x more content than the short one,
    # so a SUM would score it ~7x higher (60 vs 8 transitions). A mean
    # must stay in the same ballpark. (The END transition gives short
    # sequences a slightly larger per-transition share, so the long one
    # is allowed to be lower, but never proportionally higher.)
    assert long_score < 2 * short_score


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


def test_refit_fully_resets_state_not_accumulates():
    """Confirms the real reset-state bug is fixed: a second fit() call
    must not leave the first call's vocabulary or transition counts mixed
    in. Before the fix, self._transition_counts and self._context_totals
    (defaultdicts created once in __init__) kept growing across repeated
    fit() calls even though self.vocab_ was correctly replaced."""
    det = MarkovDetector(order=1)
    det.fit([[1, 2, 3]] * 10)
    det.fit([[5, 6, 7]] * 10)  # completely different vocabulary

    fresh = MarkovDetector(order=1).fit([[5, 6, 7]] * 10)

    assert det.vocab_ == fresh.vocab_
    assert 1 not in det.vocab_ and 2 not in det.vocab_
    assert (1,) not in det._transition_counts, "old transitions leaked across fit() calls"
    assert dict(det._context_totals) == dict(fresh._context_totals)
    np.testing.assert_allclose(det.score([[5, 6, 7]]), fresh.score([[5, 6, 7]]))


def test_scoring_does_not_mutate_fitted_state():
    """score() must be read-only, same principle as EWMA -- online
    adaptation belongs to DAWF, not to each individual detector."""
    det = MarkovDetector(order=1).fit(TRAIN)
    vocab_before = list(det.vocab_)
    totals_before = dict(det._context_totals)

    det.score([[1, 9, 2, 3]])
    det.score([SHORT_NORMAL])
    det.most_surprising_transition([1, 2, 3])

    assert det.vocab_ == vocab_before
    assert dict(det._context_totals) == totals_before


def test_unseen_penalty_is_configurable():
    det_low = MarkovDetector(order=1, unseen_penalty=5.0).fit(TRAIN)
    det_high = MarkovDetector(order=1, unseen_penalty=50.0).fit(TRAIN)
    seq = [1, 2, UNSEEN_TEMPLATE_ID, 3]
    assert det_low.score([seq])[0] < det_high.score([seq])[0]


def test_score_is_always_finite_even_for_degenerate_input():
    det = MarkovDetector(order=1).fit(TRAIN)
    weird_inputs = [
        [],
        [UNSEEN_TEMPLATE_ID] * 50,
        [1] * 10000,
    ]
    scores = det.score(weird_inputs)
    assert np.all(np.isfinite(scores)), f"non-finite score found: {scores}"


def test_fit_on_degenerate_training_data_does_not_crash():
    """All-empty or all-unseen training sequences should not crash fit()
    (vocab ends up empty, _vocab_size floors at 1 to avoid div-by-zero)."""
    det = MarkovDetector(order=1).fit([[], [], []])
    assert det.vocab_ == []
    score = det.score([[1, 2, 3]])[0]
    assert np.isfinite(score)


# -- END symbol / normal-only training (the recall fix) -------------------

COMPLETE = [1, 2, 3, 4, 5]


def test_truncated_sequence_is_flagged_with_end_symbol():
    """A sequence that just STOPS early has only normal transitions, so
    without an END token there is nothing to be surprised by."""
    train = [COMPLETE] * 100
    truncated = [1, 2, 3]

    with_end = MarkovDetector(order=1, use_end_symbol=True).fit(train)
    assert with_end.score([truncated])[0] > 3 * with_end.score([COMPLETE])[0]

    # Documents the old blind spot: without END, truncation is invisible.
    without_end = MarkovDetector(order=1, use_end_symbol=False).fit(train)
    assert without_end.score([truncated])[0] <= without_end.score([COMPLETE])[0]


def test_most_surprising_transition_reports_truncation_as_end_symbol():
    from src.detectors.markov import END_SYMBOL

    det = MarkovDetector(order=1).fit([COMPLETE] * 100)
    prev, nxt, surprisal = det.most_surprising_transition([1, 2, 3])
    assert (prev, nxt) == (3, END_SYMBOL)
    assert surprisal > 0


def test_next_event_probabilities_sum_to_one_including_end():
    det = MarkovDetector(order=1, smoothing=0.5).fit([COMPLETE] * 10 + [[1, 2, 9]] * 3)
    from src.detectors.markov import END_SYMBOL

    for context in [(1,), (2,), (5,), (-2,)]:
        total = sum(np.exp(det._log_prob(context, t)) for t in det.vocab_ + [END_SYMBOL])
        assert total == pytest.approx(1.0)


def test_token_never_seen_in_training_is_treated_as_unseen():
    """When fitting on normal-only data, a template the parser knows but
    that never appeared in normal training must get the unseen penalty,
    not a tiny smoothed probability."""
    det = MarkovDetector(order=1, unseen_penalty=20.0).fit([COMPLETE] * 50)
    score_known = det.score([COMPLETE])[0]
    score_novel = det.score([[1, 2, 99, 4, 5]])[0]  # 99 never in training
    assert score_novel > score_known
    # 99 poisons two transitions (2->99 and 99->4) at exactly the penalty.
    assert score_novel > 2 * 20.0 / 6 - 0.5


def test_most_surprising_transition_reports_original_id_for_novel_template():
    det = MarkovDetector(order=1).fit([COMPLETE] * 50)
    prev, nxt, _ = det.most_surprising_transition([1, 2, 99, 4, 5])
    assert (prev, nxt) == (2, 99)  # the real ID, not UNSEEN_TEMPLATE_ID