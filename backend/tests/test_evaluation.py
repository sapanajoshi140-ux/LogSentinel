"""Tests for src/evaluation/metrics.py and src/pipeline/evaluate.py.

Covers: correctness of the basic metrics on known small examples, safe
behaviour on empty/single-class input, proper exclusion of "Unknown"
labels rather than silently counting them as Normal, validation-based
threshold selection, and the EWMA/Markov OR & AND voting logic.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import (
    ConfusionMatrix,
    confusion_matrix,
    evaluate_predictions,
    f1_score,
    filter_unknown,
    pr_auc,
    precision,
    recall,
    select_threshold,
)
from src.pipeline.evaluate import run_evaluation


# -- confusion matrix / precision / recall / f1 ------------------------------

def test_confusion_matrix_known_example():
    preds = np.array([1, 0, 1, 1, 0])
    labels = np.array([1, 0, 0, 1, 1])
    cm = confusion_matrix(preds, labels)
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (2, 1, 1, 1)


def test_confusion_matrix_empty_input():
    cm = confusion_matrix(np.array([]), np.array([]))
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (0, 0, 0, 0)


def test_precision_recall_f1_known_values():
    cm = ConfusionMatrix(tp=2, fp=1, fn=1, tn=1)
    p, r = precision(cm), recall(cm)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f1_score(p, r) == pytest.approx(2 / 3)


def test_precision_recall_zero_when_no_positive_predictions_or_labels():
    cm = ConfusionMatrix(tp=0, fp=0, fn=0, tn=5)
    assert precision(cm) == 0.0
    assert recall(cm) == 0.0
    assert f1_score(0.0, 0.0) == 0.0


def test_n_predicted_anomalies_and_n_total():
    cm = ConfusionMatrix(tp=2, fp=3, fn=1, tn=4)
    assert cm.n_predicted_anomalies == 5
    assert cm.n_total == 10


# -- pr_auc -------------------------------------------------------------------

def test_pr_auc_computes_a_value_for_valid_input():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    auc = pr_auc(scores, labels)
    assert auc is not None
    assert 0.0 <= auc <= 1.0


def test_pr_auc_none_for_single_class():
    assert pr_auc(np.array([0.1, 0.5, 0.9]), np.array([0, 0, 0])) is None


def test_pr_auc_none_for_empty_input():
    assert pr_auc(np.array([]), np.array([])) is None


# -- threshold selection -------------------------------------------------------

def test_select_threshold_recovers_a_separating_threshold():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = select_threshold(scores, labels, metric="f1")
    preds = (scores >= t).astype(int)
    assert np.array_equal(preds, labels)


def test_select_threshold_single_class_returns_median_not_crash():
    t = select_threshold(np.array([0.1, 0.5, 0.9]), np.array([0, 0, 0]))
    assert t == pytest.approx(0.5)


def test_select_threshold_empty_input_returns_zero_not_crash():
    assert select_threshold(np.array([]), np.array([])) == 0.0


# -- filter_unknown -------------------------------------------------------------

def test_filter_unknown_drops_unknown_rows_from_every_parallel_array():
    labels_raw = ["Normal", "Anomaly", "Unknown", "Normal", "Unknown"]
    seqs = [[1], [2], [3], [4], [5]]
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.6])

    filt_labels, filt_seqs, filt_scores = filter_unknown(labels_raw, seqs, scores)

    assert filt_labels == ["Normal", "Anomaly", "Normal"]
    assert filt_seqs == [[1], [2], [4]]
    assert list(filt_scores) == [0.1, 0.9, 0.2]
    assert "Unknown" not in filt_labels


def test_filter_unknown_with_no_unknown_rows_is_a_no_op():
    labels_raw = ["Normal", "Anomaly"]
    filt_labels, = filter_unknown(labels_raw)
    assert filt_labels == labels_raw


def test_filter_unknown_all_unknown_returns_empty():
    labels_raw = ["Unknown", "Unknown"]
    seqs = [[1], [2]]
    filt_labels, filt_seqs = filter_unknown(labels_raw, seqs)
    assert filt_labels == []
    assert filt_seqs == []


# -- evaluate_predictions bundling --------------------------------------------

def test_evaluate_predictions_without_scores_has_no_pr_auc():
    preds = np.array([1, 0, 1])
    labels = np.array([1, 0, 0])
    res = evaluate_predictions(preds, labels)  # no scores given
    assert res.pr_auc is None


def test_evaluate_predictions_with_scores_has_pr_auc():
    preds = np.array([1, 0, 1, 0])
    labels = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.2, 0.6, 0.4])
    res = evaluate_predictions(preds, labels, scores=scores)
    assert res.pr_auc is not None


# -- full pipeline: run_evaluation (OR / AND voting, Unknown exclusion) -------

def _write_split_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block_id", "template_ids", "label"])
        w.writeheader()
        w.writerows(rows)


def _row(bid: str, seq: list[int], label: str) -> dict:
    return {"block_id": bid, "template_ids": json.dumps(seq), "label": label}


@pytest.fixture
def synthetic_split_dir(tmp_path) -> Path:
    """Builds a small but realistic train/val/test split with THREE kinds
    of rows, on purpose:
      - Normal blocks (the common pattern)
      - a frequency-spike anomaly (should be catchable by EWMA)
      - "Unknown"-labeled blocks, which must be excluded entirely from
        every metric, not counted as Normal.
    """
    normal = [1, 2, 3, 2]
    data_dir = tmp_path / "split"

    def gen(n_normal, n_anom, n_unknown, prefix):
        rows = [_row(f"{prefix}n{i}", normal, "Normal") for i in range(n_normal)]
        rows += [_row(f"{prefix}a{i}", [2] * 20, "Anomaly") for i in range(n_anom)]
        rows += [_row(f"{prefix}u{i}", normal, "Unknown") for i in range(n_unknown)]
        return rows

    _write_split_csv(data_dir / "train.csv", gen(200, 0, 0, "tr"))
    _write_split_csv(data_dir / "val.csv", gen(60, 6, 4, "va"))
    _write_split_csv(data_dir / "test.csv", gen(60, 6, 4, "te"))
    return data_dir


def test_run_evaluation_excludes_unknown_rows(synthetic_split_dir):
    report = run_evaluation(synthetic_split_dir)
    assert report["n_val_excluded_unknown"] == 4
    assert report["n_test_excluded_unknown"] == 4


def test_run_evaluation_returns_all_four_methods(synthetic_split_dir):
    report = run_evaluation(synthetic_split_dir)
    assert set(report["results"].keys()) == {"ewma", "markov", "or", "and"}


def test_run_evaluation_or_flags_at_least_as_many_as_either_alone(synthetic_split_dir):
    """OR-voting must never predict FEWER anomalies than either individual
    detector -- it's a logical OR of their decisions."""
    report = run_evaluation(synthetic_split_dir)
    ewma_n = report["results"]["ewma"].confusion.n_predicted_anomalies
    markov_n = report["results"]["markov"].confusion.n_predicted_anomalies
    or_n = report["results"]["or"].confusion.n_predicted_anomalies
    assert or_n >= max(ewma_n, markov_n)


def test_run_evaluation_and_flags_at_most_as_many_as_either_alone(synthetic_split_dir):
    """AND-voting must never predict MORE anomalies than either individual
    detector -- it's a logical AND of their decisions."""
    report = run_evaluation(synthetic_split_dir)
    ewma_n = report["results"]["ewma"].confusion.n_predicted_anomalies
    markov_n = report["results"]["markov"].confusion.n_predicted_anomalies
    and_n = report["results"]["and"].confusion.n_predicted_anomalies
    assert and_n <= min(ewma_n, markov_n)


def test_run_evaluation_thresholds_are_selected_independently(synthetic_split_dir):
    """EWMA and Markov scores are on totally different scales (z-scores
    vs. surprisal values) -- their selected thresholds must reflect that,
    not share one value."""
    report = run_evaluation(synthetic_split_dir)
    assert report["thresholds"]["ewma"] != report["thresholds"]["markov"]


def test_run_evaluation_or_recall_at_least_as_good_as_either_detector(synthetic_split_dir):
    """OR-voting's recall should be >= each individual detector's recall
    -- flagging on EITHER detector's say-so can only catch same-or-more
    true anomalies, never fewer."""
    report = run_evaluation(synthetic_split_dir)
    or_recall = report["results"]["or"].recall
    assert or_recall >= report["results"]["ewma"].recall - 1e-9
    assert or_recall >= report["results"]["markov"].recall - 1e-9