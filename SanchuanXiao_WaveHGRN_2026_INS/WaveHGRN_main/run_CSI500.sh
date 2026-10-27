#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

cd "$PROJECT_DIR"
exec "$PYTHON" training/mytrain.py \
  --period CSI500 \
  -length 96 \
  -feature 9 \
  -n_class 2 \
  -epoch 200 \
  -batch_size 32 \
  --tem_dim 96 \
  --hidden 96 \
  -hyper_edge 15 \
  -dropout 0.35 \
  --lr 5e-4 \
  --scale_num 2 \
  --path_num 6 \
  --mem_dim 96 \
  --concentration_weight 0.5
