#!/usr/bin/env bash
set -euo pipefail
: "${BEST276_ROOT:=.}"
: "${BEST276_RUN_ROOT:?Set BEST276_RUN_ROOT to the prepared run directory}"
: "${BEST276_SOURCE_DATA:?Set BEST276_SOURCE_DATA to the local dataset path}"
: "${BEST276_CONFIG:=$BEST276_ROOT/configs/main_experiment.yaml}"
PYTHONPATH="$BEST276_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m best_memory.answering.pipeline \
  --config "$BEST276_CONFIG" \
  --run-root "$BEST276_RUN_ROOT" \
  --input "$BEST276_SOURCE_DATA"
