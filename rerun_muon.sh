#!/usr/bin/env bash
# Rerun all Muon experiments for 2, 4, 6 layers.
#   muon        : Muon on 2D hidden weights, AdamW on embeddings + 1D weights
#   muon_all2d  : Muon on every 2D weight, AdamW on 1D weights
# lr(L) = lr_ref * sqrt(4 / L)
set -euo pipefail

PYTHON=/home/jovyan/.mlspace/envs/yusupov_mrgsrec/bin/python
cd "$(dirname "$0")"
mkdir -p logs

for L in 2 4 6; do
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" run_model.py \
        -cn beauty_sasrec_muon \
        model.num_layers="$L" \
        hydra.run.dir="outputs/muon_l${L}" \
        hydra.job.chdir=false \
        > "logs/muon_l${L}.log" 2>&1 &
done

for L in 2 4 6; do
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" run_model.py \
        -cn beauty_sasrec_muon2d \
        model.num_layers="$L" \
        hydra.run.dir="outputs/muon_all2d_l${L}" \
        hydra.job.chdir=false \
        > "logs/muon_all2d_l${L}.log" 2>&1 &
done

wait
