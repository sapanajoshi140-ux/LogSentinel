# precision/recall/F1, PR curves
"""
Reusable evaluation functions shared by every detector and by the
threshold-voting pipeline (src/pipeline/evaluate.py).

WHY "Unknown" LABELS NEED SPECIAL HANDLING
    A block with no entry in anomaly_label.csv is not confirmed Normal --
    it's simply unlabeled. src/features/loading.py's basic load_split_csv()
    collapses "Unknown" into the same 0 as "Normal" for convenience, but
    counting an unconfirmed block as a true negative would understate the
    false-positive rate and overstate precision. Every function here that
    computes precision/recall/F1/confusion-matrix takes labels that are
    assumed to ALREADY exclude Unknown rows -- use filter_unknown() first.

THRESHOLD SELECTION METHODOLOGY (why this file exists as its own module)
    A detector's raw score (an EWMA z-score, a Markov surprisal value) is
    not itself a prediction -- turning it into "anomaly" or "not" requires
    a threshold, and that threshold must be chosen on the VALIDATION split,
    then FROZEN before touching the test split. Choosing it on test data
    would let test-set outcomes leak into the decision that produces
    test-set metrics, which is exactly the kind of leakage the rest of
    this project (src/parsing/leakage_free_parser.py, chronological
    splitting) is built to avoid elsewhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.metrics import average_precision_score
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only without sklearn installed
    _HAS_SKLEARN = False


@dataclass
class ConfusionMatrix:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n_predicted_anomalies(self) -> int:
        return self.tp + self.fp

    @property
    def n_total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


@dataclass
class EvaluationResult:
    """Everything reported for one detector or one voting method, in one
    place -- src/pipeline/evaluate.py builds one of these per method
    (EWMA-only, Markov-only, OR, AND) so they can be printed/compared
    uniformly."""
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix
    pr_auc: float | None = None  # None when not applicable (e.g. OR/AND voting has no continuous score) or sklearn unavailable

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "tp": self.confusion.tp,
            "fp": self.confusion.fp,
            "fn": self.confusion.fn,
            "tn": self.confusion.tn,
            "n_predicted_anomalies": self.confusion.n_predicted_anomalies,
        }


def filter_unknown(
    labels_raw: list[str],
    *parallel_arrays,
) -> tuple:
    """Drop every position where labels_raw[i] == "Unknown" from
    labels_raw and from each array in parallel_arrays (they must all be
    the same length and in the same order -- e.g. sequences, predictions,
    scores, ids). Returns a tuple: (filtered_labels_raw, *filtered_parallel_arrays),
    in the same order they were given.

    This is the ONLY place "Unknown" rows are dropped -- everything else
    in this module assumes it has already happened.
    """
    keep = [i for i, lbl in enumerate(labels_raw) if lbl != "Unknown"]
    filtered_labels = [labels_raw[i] for i in keep]
    filtered_arrays = tuple(
        [arr[i] for i in keep] if not isinstance(arr, np.ndarray) else arr[keep]
        for arr in parallel_arrays
    )
    return (filtered_labels, *filtered_arrays)


def confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> ConfusionMatrix:
    """preds and labels are both 0/1 arrays (or array-likes), same length.
    Empty input returns an all-zero matrix rather than raising."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if len(preds) == 0:
        return ConfusionMatrix(tp=0, fp=0, fn=0, tn=0)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def precision(cm: ConfusionMatrix) -> float:
    denom = cm.tp + cm.fp
    return cm.tp / denom if denom > 0 else 0.0


def recall(cm: ConfusionMatrix) -> float:
    denom = cm.tp + cm.fn
    return cm.tp / denom if denom > 0 else 0.0


def f1_score(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the precision-recall curve, from continuous scores (not
    thresholded predictions). Returns None -- not 0.0, which would look
    like a real, bad score -- when it genuinely cannot be computed:
    sklearn isn't installed, there's no data, or labels contain only one
    class (PR-AUC is undefined with no positive examples, or trivially 1.0
    with no negative examples -- neither is a meaningful number)."""
    if not _HAS_SKLEARN:
        return None
    labels = np.asarray(labels)
    if len(labels) == 0:
        return None
    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, np.asarray(scores)))


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    n_candidates: int = 200,
) -> float:
    """Search over candidate thresholds and return the one maximizing
    `metric` ("f1", "precision", or "recall") on the given (scores,
    labels) -- intended to be called on the VALIDATION split only; see
    module docstring for why. Candidates are `n_candidates` evenly spaced
    quantiles of the observed scores, not every unique score value, so
    this stays fast even with hundreds of thousands of validation rows.

    Safety behaviour, rather than crashing:
      - Empty input: returns 0.0 (an arbitrary but harmless default; there
        is nothing to select a threshold FROM).
      - Only one class present in labels: F1/precision/recall are not
        meaningful for threshold selection in this case (e.g. with zero
        positives, every threshold gives recall=0/0). Returns the median
        observed score as a sane default rather than raising, since a
        validation split with only one class is unusual but not something
        that should crash an evaluation run.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(scores) == 0:
        return 0.0
    if len(np.unique(labels)) < 2:
        return float(np.median(scores))

    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, n_candidates)))
    best_threshold, best_value = candidates[0], -1.0

    for t in candidates:
        preds = (scores >= t).astype(int)
        cm = confusion_matrix(preds, labels)
        p, r = precision(cm), recall(cm)
        value = {"f1": f1_score(p, r), "precision": p, "recall": r}[metric]
        if value > best_value:
            best_threshold, best_value = t, value

    return float(best_threshold)


def evaluate_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
) -> EvaluationResult:
    """Bundle confusion matrix + precision/recall/F1 (+ PR-AUC if
    continuous `scores` are given) into one EvaluationResult. `scores` is
    optional and should be omitted for methods with no single continuous
    score of their own (e.g. OR/AND voting between two detectors' already-
    thresholded predictions) -- PR-AUC will simply be None in that case."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cm = confusion_matrix(preds, labels)
    p, r = precision(cm), recall(cm)
    auc = pr_auc(scores, labels) if scores is not None else None
    return EvaluationResult(precision=p, recall=r, f1=f1_score(p, r), confusion=cm, pr_auc=auc)