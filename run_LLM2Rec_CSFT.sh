#!/usr/bin/env bash
set -euo pipefail

# First stage of LLM2Rec training -- Collaborative Supervised Fine-Tuning (CSFT).

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

model_path="${MODEL_PATH:-./models/Qwen2-0.5B}"
gpu_ids="${GPU_IDS:-0,1}"
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
nproc_per_node=$(awk -F',' '{print NF}' <<< "${gpu_ids}")

for category in "AmazonMix-6"
do
    train_file=$(ls -f ./data/${category}/5-core/train/${category}*.csv)
    eval_file=$(ls -f ./data/${category}/5-core/valid/${category}*.csv)
    output_dir="./output/csft/Qwen2-0.5B-${category}"
    echo ${train_file} ${eval_file}

    resume_args=()
    if [ -n "${resume_from_checkpoint}" ]; then
        resume_args=(--resume_from_checkpoint "${resume_from_checkpoint}")
    fi

    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun --master_port=25649 --nproc_per_node ${nproc_per_node} \
        ./llm2rec/run_csft.py \
        --base_model ${model_path} \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir ${output_dir} \
        --wandb_run_name Qwen2-0.5B-CSFT-${category} \
        --category ${category} \
        --train_from_scratch False \
        --use_lora False \
        "${resume_args[@]}"

    cp ${model_path}/*token* ${output_dir}/
    # Also copy tokenizer to the last checkpoint
    latest_ckpt=$(ls -d ${output_dir}/checkpoint-* | sort -V | tail -n 1)
    if [ -n "${latest_ckpt}" ]; then
        cp ${model_path}/*token* ${latest_ckpt}/
    fi
done
