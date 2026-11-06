#!/usr/bin/env bash
# Depth-scaled LR experiments: lr(L) = lr_ref * sqrt(4 / L).
# AdamW on all parameters vs Muon on 2D weights + AdamW on the rest.
set -euo pipefail

PYTHON=/home/jovyan/.mlspace/envs/yusupov_mrgsrec/bin/python
cd "$(dirname "$0")"

mkdir -p logs

for L in 2 4 6; do
    CUDA_VISIBLE_DEVICES=3 "$PYTHON" run_model.py \
        -cn beauty_sasrec_adamw \
        model.num_layers="$L" \
        hydra.run.dir="outputs/adamw_l${L}" \
        hydra.job.chdir=false \
        > "logs/adamw_l${L}.log" 2>&1 &
done

for L in 2 4 6; do
    CUDA_VISIBLE_DEVICES=6 "$PYTHON" run_model.py \
        -cn beauty_sasrec_muon \
        model.num_layers="$L" \
        hydra.run.dir="outputs/muon_l${L}" \
        hydra.job.chdir=false \
        > "logs/muon_l${L}.log" 2>&1 &
done

wait
