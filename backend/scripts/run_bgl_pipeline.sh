#!/usr/bin/env bash
# End-to-end BGL pipeline: parse -> sliding-window split -> train + evaluate.
# Mirrors run_hdfs_pipeline.sh but uses the sliding-window splitter, since BGL
# has no block ID to group by (see src/splitting/bgl_split.py docstring).
#
#   1. src/parsing/leakage_free_parser.py   (raw log -> parsed_leakage_free.csv)
#   2. src/splitting/bgl_split.py           (parsed csv -> train/val/test.csv)
#   3. src/pipeline/evaluate.py             (train/val/test.csv -> metrics)
#
# Usage (from inside backend/, venv active):
#   bash scripts/run_bgl_pipeline.sh
#
# Requires data/raw/BGL.log to already exist (loghub BGL:
# https://github.com/logpai/loghub). This script does not download data.
#
# NOTE: config/bgl.yaml currently has `dataset: hdfs` and HDFS paths --
# that looks like a copy-paste leftover from hdfs.yaml. Fix it (dataset: bgl,
# paths pointing at BGL.log / data/processed/bgl / artifacts/bgl / results/bgl)
# before citing "config-driven runs" tomorrow, or a run with `--config
# config/bgl.yaml` will silently try to load HDFS.log.

set -euo pipefail

RAW_LOG="${RAW_LOG:-data/raw/BGL.log}"
PARSED_DIR="${PARSED_DIR:-data/processed/bgl_parsed}"
SPLIT_DIR="${SPLIT_DIR:-data/processed/bgl_split}"
RESULTS_DIR="${RESULTS_DIR:-results/bgl}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
VAL_FRAC="${VAL_FRAC:-0.15}"
WINDOW_SIZE="${WINDOW_SIZE:-20}"
STRIDE="${STRIDE:-20}"
EWMA_ALPHA="${EWMA_ALPHA:-0.3}"
MARKOV_ORDER="${MARKOV_ORDER:-1}"

if [[ ! -f "$RAW_LOG" ]]; then
  echo "ERROR: raw log not found at $RAW_LOG"
  echo "Download BGL from loghub (https://github.com/logpai/loghub)"
  echo "and place BGL.log under data/raw/."
  exit 1
fi

echo "=== [1/3] Leakage-free Drain3 parsing (train-only template learning) ==="
python -m src.parsing.leakage_free_parser \
  --input "$RAW_LOG" \
  --dataset bgl \
  --output-dir "$PARSED_DIR" \
  --train-frac "$TRAIN_FRAC" \
  --val-frac "$VAL_FRAC"

echo
echo "=== [2/3] Sliding-window sequences (size=$WINDOW_SIZE, stride=$STRIDE) + chronological split ==="
python -m src.splitting.bgl_split \
  --parsed "$PARSED_DIR/parsed_leakage_free.csv" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
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
echo "BGL is ~3% anomalous and imbalanced -- expect noisier F1 than HDFS."