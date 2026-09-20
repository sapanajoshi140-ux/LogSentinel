
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


def load_split_csv(
    path: str | Path,
) -> tuple[list[list[int]], list[int], list[str]]:
    """Returns (sequences, labels, ids).

    sequences: one list[int] of template IDs per row (UNSEEN_TEMPLATE_ID,
        -1, included as-is — detectors decide how to treat it).
    labels: 1 for "Anomaly", 0 for anything else (including "Unknown" —
        blocks with no entry in anomaly_label.csv). Treating "Unknown" as
        0 is a scoring convenience only; anything that computes precision/
        recall/F1 downstream (src/evaluation/metrics.py) should filter
        "Unknown" rows out rather than silently counting them as Normal.
    ids: the block_id (HDFS) or start_line_no (BGL) for each row, in the
        same order as sequences/labels, so results can be traced back to
        a specific block or window.
    """
    path = Path(path)
    sequences, labels, ids = [], [], []

    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return sequences, labels, ids

        id_col = "block_id" if "block_id" in reader.fieldnames else "start_line_no"

        for row in reader:
            template_ids = [int(t) for t in json.loads(row["template_ids"])]
            sequences.append(template_ids)
            labels.append(1 if row["label"] == "Anomaly" else 0)
            ids.append(row[id_col])

    return sequences, labels, ids
