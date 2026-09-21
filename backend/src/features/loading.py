"""
Load the train/val/test CSVs produced by src/splitting/hdfs_split.py and
src/splitting/bgl_split.py into plain Python structures every detector can
consume, without each detector reimplementing CSV/JSON parsing.

Both splitters write the same shape — an id column, a JSON-encoded list of
template IDs, and a label — just with a different id column name
(block_id for HDFS, start_line_no for BGL). This auto-detects which.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_rows(path: str | Path):
    path = Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        id_col = "block_id" if "block_id" in reader.fieldnames else "start_line_no"
        for row in reader:
            template_ids = [int(t) for t in json.loads(row["template_ids"])]
            yield template_ids, row["label"], row[id_col]


def load_split_csv(
    path: str | Path,
) -> tuple[list[list[int]], list[int], list[str]]:
    """Returns (sequences, labels, ids).

    sequences: one list[int] of template IDs per row (UNSEEN_TEMPLATE_ID,
        -1, included as-is — detectors decide how to treat it).
    labels: 1 for "Anomaly", 0 for anything else (including "Unknown" —
        blocks with no entry in anomaly_label.csv). Treating "Unknown" as
        0 is a scoring convenience only, and loses the distinction between
        a real Normal and an Unknown row. For evaluation code that needs
        to EXCLUDE "Unknown" rows properly (src/evaluation/metrics.py
        does), use load_split_csv_with_raw_labels() instead, which keeps
        the original label string around. This function is kept exactly
        as-is for backward compatibility with scripts/try_ewma.py,
        scripts/try_markov.py, and scripts/try_dawf.py, which only ever
        needed a binary label.
    ids: the block_id (HDFS) or start_line_no (BGL) for each row, in the
        same order as sequences/labels, so results can be traced back to
        a specific block or window.
    """
    sequences, labels, ids = [], [], []
    for template_ids, raw_label, id_ in _read_rows(path):
        sequences.append(template_ids)
        labels.append(1 if raw_label == "Anomaly" else 0)
        ids.append(id_)
    return sequences, labels, ids


def load_split_csv_with_raw_labels(
    path: str | Path,
) -> tuple[list[list[int]], list[int], list[str], list[str]]:
    """Same as load_split_csv(), but ALSO returns the original label
    string per row (raw_labels), so callers can tell a real "Normal" apart
    from an "Unknown" (a block with no entry in anomaly_label.csv) --
    load_split_csv()'s binary labels collapse both to 0, which is fine for
    a quick score sanity-check but wrong for anything computing precision/
    recall/F1, where an "Unknown" row should be EXCLUDED, not silently
    counted as a confirmed Normal.

    Returns (sequences, labels, ids, raw_labels). labels is the same
    binary 1/0 encoding as load_split_csv(); raw_labels is the exact
    string from the CSV ("Normal", "Anomaly", or "Unknown"), same order.
    """
    sequences, labels, ids, raw_labels = [], [], [], []
    for template_ids, raw_label, id_ in _read_rows(path):
        sequences.append(template_ids)
        labels.append(1 if raw_label == "Anomaly" else 0)
        ids.append(id_)
        raw_labels.append(raw_label)
    return sequences, labels, ids, raw_labels


def normal_only(sequences: list[list[int]], raw_labels: list[str]) -> list[list[int]]:
    """Keep only sequences whose raw label is exactly "Normal".

    Used to fit detectors that model NORMAL behaviour (Markov). Both
    "Anomaly" rows (their odd transitions would be learned as normal) and
    "Unknown" rows (not confirmed normal) are dropped. Needs the raw label
    strings from load_split_csv_with_raw_labels(), because the binary
    labels from load_split_csv() collapse "Unknown" into 0.
    """
    if len(sequences) != len(raw_labels):
        raise ValueError("sequences and raw_labels must be the same length")
    return [seq for seq, lbl in zip(sequences, raw_labels) if lbl == "Normal"]