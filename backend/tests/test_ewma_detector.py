"""Tests for the EWMA detector and its supporting pieces.

These check the properties that actually matter for using EWMA inside
DAWF: normal sequences score low, genuinely anomalous ones score high,
fit() freezes state rather than adapting during score(), and unseen
templates are handled without special-casing.
"""

import numpy as np
import pytest

from src.detectors.ewma import EWMADetector
from src.features.counts import UNSEEN_TEMPLATE_ID, EventCountVectorizer

NORMAL_SEQ = [1, 2, 3, 2]
TRAIN_SEQUENCES = [NORMAL_SEQ] * 50


# -- EventCountVectorizer --------------------------------------------------

def test_vectorizer_counts_known_templates():
    vec = EventCountVectorizer().fit([[1, 2, 2, 3]])
    X = vec.transform([[1, 2, 2, 3]])
    assert X.shape == (1, len(vec.vocab_) + 1)
    assert X[0, vec.vocab_.index(2)] == 2


def test_vectorizer_routes_unknown_to_trailing_bucket():
    vec = EventCountVectorizer().fit([[1, 2, 3]])
    X = vec.transform([[1, 99, 99]])  # 99 never seen in training
    assert X[0, -1] == 2  # both 99s land in the unknown bucket


def test_vectorizer_routes_unseen_marker_to_trailing_bucket():
    vec = EventCountVectorizer().fit([[1, 2, 3]])
    X = vec.transform([[1, UNSEEN_TEMPLATE_ID, UNSEEN_TEMPLATE_ID]])
    assert X[0, -1] == 2


def test_vectorizer_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        EventCountVectorizer().transform([[1, 2]])


# -- EWMADetector -----------------------------------------------------------

def test_fit_requires_at_least_one_sequence():
    with pytest.raises(ValueError):
        EWMADetector().fit([])


def test_score_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        EWMADetector().score([NORMAL_SEQ])


def test_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        EWMADetector(alpha=0.0)
    with pytest.raises(ValueError):
        EWMADetector(alpha=1.5)


def test_repeated_normal_sequence_scores_near_zero():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    scores = det.score([NORMAL_SEQ, NORMAL_SEQ])
    # EWMA converges geometrically toward the true mean, not exactly to it,
    # so this checks "negligibly small relative to a real anomaly", not
    # "zero to float precision".
    assert np.all(scores < 0.01)


def test_novel_template_scores_far_higher_than_normal():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    normal_score = det.score([NORMAL_SEQ])[0]
    novel_score = det.score([[1, 2, 99, 99, 99]])[0]  # 99 never in training
    assert novel_score > normal_score * 10


def test_frequency_spike_scores_higher_than_normal():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    normal_score = det.score([NORMAL_SEQ])[0]
    spike_score = det.score([[2] * 20])[0]
    assert spike_score > normal_score


def test_unseen_marker_in_test_sequence_scores_high():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    score = det.score([[1, 2, UNSEEN_TEMPLATE_ID]])[0]
    normal_score = det.score([NORMAL_SEQ])[0]
    assert score > normal_score


def test_fit_does_not_mutate_during_score():
    """score() must be read-only -- online adaptation belongs to DAWF,
    not to each individual detector. See module docstring for why."""
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    mean_before = det._mean.copy()

    det.score([[1, 2, 99, 99]])
    det.score([NORMAL_SEQ])

    assert np.array_equal(mean_before, det._mean)


def test_most_deviant_template_identifies_the_spiking_template():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    template_id, z = det.most_deviant_template([2] * 20)
    assert template_id == 2
    assert z > 0


def test_most_deviant_template_returns_none_for_unknown_bucket():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    template_id, z = det.most_deviant_template([1, 2, 99, 99, 99])
    assert template_id is None


def test_score_returns_empty_array_for_empty_input():
    det = EWMADetector(alpha=0.3).fit(TRAIN_SEQUENCES)
    assert len(det.score([])) == 0


def test_long_normal_block_does_not_outscore_true_anomaly():
    """Regression test for a real bug found on HDFS_v1: a Normal block that
    is simply LONGER than average (more replication events, no anomaly)
    produced a raw count far above the training mean purely from length,
    scoring ~200,000 -- dwarfing every true anomaly's score (~3,000).
    Length-normalized counts (the default) must fix this: a long block with
    normal PROPORTIONS should score low, well below a short block whose
    proportions are actually skewed.
    """
    # Training: short "normal" blocks, template 2 always ~50% of the block.
    train = [[1, 2, 3, 2]] * 100

    # A long block, 40 lines, but the SAME 1:2:3:2-style proportions
    # repeated 10x over -- longer, not anomalous.
    long_normal = [1, 2, 3, 2] * 10

    # A short block where the proportions are genuinely skewed (template 2
    # dominates far more than the learned normal ratio).
    short_skewed = [2, 2, 2, 2, 2, 2, 2, 2]

    det = EWMADetector(alpha=0.3, normalize=True).fit(train)
    long_normal_score = det.score([long_normal])[0]
    short_skewed_score = det.score([short_skewed])[0]

    assert short_skewed_score > long_normal_score, (
        f"length-normalization regression: a merely-longer normal block "
        f"(score={long_normal_score:.1f}) outscored a genuinely skewed "
        f"short block (score={short_skewed_score:.1f})"
    )


def test_raw_counts_mode_is_still_available_but_opt_in():
    """normalize=False should reproduce the old (buggy-for-long-blocks)
    raw-count behaviour, for anyone who explicitly wants it."""
    det = EWMADetector(alpha=0.3, normalize=False).fit(TRAIN_SEQUENCES)
    assert det.vectorizer.normalize is False
