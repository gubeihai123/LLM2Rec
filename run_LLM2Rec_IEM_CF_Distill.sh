#!/usr/bin/env bash
set -euo pipefail

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

gpu_id="${GPU_ID:-4}"
python_bin="${PYTHON_BIN:-python}"
teacher_item_emb_path="${TEACHER_ITEM_EMB_PATH:?Set TEACHER_ITEM_EMB_PATH to a SASRec teacher .pt tensor.}"
item_titles_path="${ITEM_TITLES_PATH:-data/AmazonMix-6/5-core/info/item_titles.txt}"
sequence_path="${SEQUENCE_PATH:-data/AmazonMix-6/5-core/sequential_data.txt}"
cf_distill_path="output/cf_distill/AmazonMix6_top16.pt"

precompute_args=(
    --teacher_item_emb_path "${teacher_item_emb_path}" \
    --item_titles_path "${item_titles_path}" \
    --output_path "${cf_distill_path}" \
    --top_k 16 \
    --teacher_tau 0.05 \
    --use_reliability true \
    --support_cap 100
)
if [ -n "${sequence_path}" ]; then
    precompute_args+=(--sequence_path "${sequence_path}")
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" ./llm2rec/precompute_cf_distill.py \
    "${precompute_args[@]}" \
    --device cuda

CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" \
    ./llm2rec/run_unsupervised_SimCSE.py \
    ./llm2rec/train_simcse_cfkd_uncalibrated_config.json

CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" \
    ./llm2rec/run_unsupervised_SimCSE.py \
    ./llm2rec/train_simcse_cfkd_reliability_config.json
