#!/usr/bin/env bash
# End-to-end HDFS pipeline: parse -> split -> train + evaluate (EWMA, Markov,
# OR-vote, AND-vote), exactly following the methodology documented at the top
# of src/pipeline/evaluate.py (fit on train, threshold on val, freeze, report
# on test).
#
# This fills in the empty scripts/run_hdfs_pipeline.sh stub by chaining the
# three commands that were already implemented and tested, but never wired
# together into one reproducible entry point:
#   1. src/parsing/leakage_free_parser.py   (raw log -> parsed_leakage_free.csv)
#   2. src/splitting/hdfs_split.py          (parsed csv -> train/val/test.csv)
#   3. src/pipeline/evaluate.py             (train/val/test.csv -> metrics)
#
# Usage (from inside backend/, with the venv from requirements.txt active):
#   bash scripts/run_hdfs_pipeline.sh
#
# Every path/hyperparameter can be overridden via environment variables so
# this matches whatever is in config/hdfs.yaml, e.g.:
#   TRAIN_FRAC=0.8 EWMA_ALPHA=0.2 bash scripts/run_hdfs_pipeline.sh
#
# Requires data/raw/HDFS.log and data/raw/anomaly_label.csv to already exist
# (loghub HDFS_v1: https://github.com/logpai/loghub). This script does not
# download data itself.

set -euo pipefail

RAW_LOG="${RAW_LOG:-data/raw/hdfs.log}"
LABELS="${LABELS:-data/raw/anomaly_label.csv}"
PARSED_DIR="${PARSED_DIR:-data/processed/hdfs_parsed}"
SPLIT_DIR="${SPLIT_DIR:-data/processed/hdfs_split}"
RESULTS_DIR="${RESULTS_DIR:-results/hdfs}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
VAL_FRAC="${VAL_FRAC:-0.15}"
EWMA_ALPHA="${EWMA_ALPHA:-0.3}"
MARKOV_ORDER="${MARKOV_ORDER:-1}"

if [[ ! -f "$RAW_LOG" ]]; then
  echo "ERROR: raw log not found at $RAW_LOG"
  echo "Download HDFS_v1 from loghub (https://github.com/logpai/loghub)"
  echo "and place HDFS.log + anomaly_label.csv under data/raw/."
  exit 1
fi
if [[ ! -f "$LABELS" ]]; then
  echo "ERROR: labels file not found at $LABELS"
  exit 1
fi

echo "=== [1/3] Leakage-free Drain3 parsing (train-only template learning) ==="
python -m src.parsing.leakage_free_parser \
  --input "$RAW_LOG" \
  --dataset hdfs \
  --output-dir "$PARSED_DIR" \
  --train-frac "$TRAIN_FRAC" \
  --val-frac "$VAL_FRAC"

echo
echo "=== [2/3] Grouping into per-block sequences + train/val/test split ==="
python -m src.splitting.hdfs_split \
  --parsed "$PARSED_DIR/parsed_leakage_free.csv" \
  --labels "$LABELS" \
  --output-dir "$SPLIT_DIR" \
  --train-frac "$TRAIN_FRAC" \
  --val-frac "$VAL_FRAC"

echo
echo "=== [3/3] Fit EWMA + Markov on train, threshold on val, report on frozen test ==="
mkdir -p "$RESULTS_DIR"
python -m src.pipeline.evaluate \
  --data-dir "$SPLIT_DIR" \
  --ewma-alpha "$EWMA_ALPHA" \
  --markov-order "$MARKOV_ORDER" \
  --output-json "$RESULTS_DIR/mid_progress_results.json"

echo
echo "Done. Results saved to $RESULTS_DIR/mid_progress_results.json"
echo "(Calibrated DAWF fusion of the same two detectors: python scripts/try_dawf.py)"