"""Tests for DAWF fusion.

test_favors_the_actually_better_detector_on_real_imbalanced_data is the
important one: it reproduces the real bug found on actual HDFS data
(precision/recall numbers below are the real ones), where a naive
weighting scheme favored a cautious, low-recall detector (EWMA) over one
that was genuinely better overall (Markov), purely because of class
imbalance. This locks in that the fix actually works, using the real
numbers, not just a friendly synthetic case.
"""

import numpy as np
import pytest

from src.fusion.dawf import DAWF


def test_rejects_invalid_eta():
    with pytest.raises(ValueError):
        DAWF(["a", "b"], eta=0)
    with pytest.raises(ValueError):
        DAWF(["a", "b"], eta=-1)


def test_rejects_invalid_min_weight():
    with pytest.raises(ValueError):
        DAWF(["a", "b"], min_weight=0.6)  # >= 1/2, leaves no room


def test_weights_start_equal():
    dawf = DAWF(["ewma", "markov", "bilstm"])
    weights = dawf.current_weights()
    assert weights["ewma"] == pytest.approx(1 / 3)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_always_sum_to_one_after_update():
    dawf = DAWF(["a", "b"])
    dawf.update({"a": 0.9, "b": 0.1}, true_label=0, threshold=0.5)
    assert sum(dawf.current_weights().values()) == pytest.approx(1.0)


def test_correct_prediction_is_not_penalized():
    dawf = DAWF(["a", "b"])
    before = dawf.current_weights()
    dawf.update({"a": 0.1, "b": 0.1}, true_label=0, threshold=0.5)  # both correctly say "normal"
    after = dawf.current_weights()
    assert before == after


def test_wrong_detector_loses_weight_relative_to_correct_one():
    dawf = DAWF(["a", "b"])
    # "a" wrongly says anomaly, "b" correctly says normal
    dawf.update({"a": 0.9, "b": 0.1}, true_label=0, threshold=0.5)
    weights = dawf.current_weights()
    assert weights["b"] > weights["a"]


def test_min_weight_floor_is_respected_after_many_mistakes():
    dawf = DAWF(["a", "b"], eta=0.5, min_weight=0.05)
    for _ in range(200):
        dawf.update({"a": 0.9, "b": 0.1}, true_label=0, threshold=0.5)  # "a" always wrong
    assert dawf.current_weights()["a"] >= 0.05 - 1e-9


def test_weights_never_nan_or_infinite_after_many_updates():
    """Regression test: a naive multiplicative implementation underflows
    to exactly 0.0 for both detectors after enough steps, which can turn
    a subsequent renormalization into NaN. This must stay finite over a
    realistic number of updates."""
    dawf = DAWF(["a", "b"])
    rng = np.random.RandomState(0)
    for _ in range(100_000):
        # "a" is right 99% of the time, "b" is right 60% of the time
        a_correct = rng.rand() < 0.99
        b_correct = rng.rand() < 0.60
        label = int(rng.rand() < 0.5)
        a_score = 0.9 if (label == (1 if a_correct else 0)) else 0.1
        scores = {
            "a": 0.9 if a_correct == (label == 1) else 0.1,
            "b": 0.9 if b_correct == (label == 1) else 0.1,
        }
        dawf.update(scores, true_label=label, threshold=0.5)

    weights = dawf.current_weights()
    assert all(np.isfinite(v) for v in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weight_trajectory_records_every_update():
    dawf = DAWF(["a", "b"])
    for _ in range(10):
        dawf.update({"a": 0.5, "b": 0.5}, true_label=0, threshold=0.5)
    assert len(dawf.weight_trajectory()) == 10


def test_predict_returns_weighted_average_and_threshold_decision():
    dawf = DAWF(["a", "b"])  # starts 50/50
    fused, pred = dawf.predict({"a": 0.8, "b": 0.2}, threshold=0.5)
    assert fused == pytest.approx(0.5)
    assert pred == 1


def test_favors_the_actually_better_detector_on_real_imbalanced_data():
    """Regression test using the REAL precision/recall/imbalance numbers
    found on actual HDFS_v1 data:
        EWMA:   recall=0.315, precision=0.989 (cautious, low recall)
        Markov: recall=0.485, precision=0.548 (better F1 overall)
        anomaly rate: 1116 / 86260 (~1.3%)

    A naive symmetric weighting scheme favored EWMA here purely because
    class imbalance made "rarely flagging anything" look good by default.
    DAWF must favor Markov -- the detector that is actually better.
    """
    rng = np.random.RandomState(0)
    n = 20_000
    anomaly_rate = 1116 / 86260
    true_labels = (rng.rand(n) < anomaly_rate).astype(int)
    n_anom, n_norm = true_labels.sum(), n - true_labels.sum()

    ewma_recall, ewma_precision = 0.315, 0.989
    mk_recall, mk_precision = 0.485, 0.548

    ewma_fp_rate = (ewma_recall * n_anom) * (1 - ewma_precision) / ewma_precision / n_norm
    mk_fp_rate = (mk_recall * n_anom) * (1 - mk_precision) / mk_precision / n_norm

    dawf = DAWF(["ewma", "markov"])
    for label in true_labels:
        if label == 1:
            ewma_p = 0.6 if rng.rand() < ewma_recall else 0.1
            mk_p = 0.6 if rng.rand() < mk_recall else 0.1
        else:
            ewma_p = 0.6 if rng.rand() < ewma_fp_rate else 0.1
            mk_p = 0.6 if rng.rand() < mk_fp_rate else 0.1
        scores = {"ewma": ewma_p, "markov": mk_p}
        dawf.update(scores, true_label=int(label), threshold=0.5)

    weights = dawf.current_weights()
    assert weights["markov"] > weights["ewma"], (
        f"expected Markov (the actually better detector) to win, got {weights}"
    )