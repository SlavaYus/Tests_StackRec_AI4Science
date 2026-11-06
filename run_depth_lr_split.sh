#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/jovyan/.mlspace/envs/yusupov_mrgsrec/bin/python
cd "$(dirname "$0")"

mkdir -p logs

# Muon on 2D hidden weights only; embeddings + 1D -> AdamW. lr(L)=lr_ref*sqrt(4/L).
for L in 2 4 6; do
    CUDA_VISIBLE_DEVICES=3 "$PYTHON" run_model.py \
        -cn beauty_sasrec_muon2d \
        model.num_layers="$L" \
        hydra.run.dir="outputs/muon_l${L}" \
        hydra.job.chdir=false \
        > "logs/muon_l${L}.log" 2>&1 &
done

wait
