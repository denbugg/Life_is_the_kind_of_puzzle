#!/usr/bin/env bash
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
STAGE_DIR="$(mktemp -d /tmp/kaggle-e13-border-encoder.XXXXXX)"
cp kaggle_train_border_encoder.py "$STAGE_DIR/kaggle_train_border_encoder.py"
cp kernel-metadata-border-encoder.json "$STAGE_DIR/kernel-metadata.json"
"$KAGGLE_BIN" kernels push -p "$STAGE_DIR" -t 3600 --accelerator NvidiaTeslaT4
