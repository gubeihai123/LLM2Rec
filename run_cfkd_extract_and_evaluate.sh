#!/usr/bin/env bash
set -euo pipefail

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

python_bin="${PYTHON_BIN:-/HDDDATA/wby/LLM2Rec/.venv/bin/python}"
datasets="${DATASETS:-Games_5core Arts_5core Movies_5core Sports_5core Baby_5core Goodreads}"
gpu_ids="${GPU_IDS:-3,4,0}"
extraction_method="${EXTRACTION_METHOD:-title}"
log_dir="${LOG_DIR:-run_logs/cfkd_eval_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${log_dir}"

models_list=(
    "Qwen2-0.5B-CFKD-uncalibrated ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CFKD-uncalibrated/checkpoint-1000"
    "Qwen2-0.5B-CFKD-reliability ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CFKD-reliability/checkpoint-1000"
)

IFS=' ' read -r -a dataset_array <<< "${datasets}"
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"

run_dataset() {
    local dataset="$1"
    local cuda_device="$2"

    mkdir -p "./item_info/${dataset}"
    echo "[$(date '+%F %T')] Start dataset=${dataset}, gpu=${cuda_device}"

    for model_setting in "${models_list[@]}"; do
        local save_info
        local model_path
        save_info=$(awk '{print $1}' <<< "${model_setting}")
        model_path=$(awk '{print $2}' <<< "${model_setting}")

        if [ ! -d "${model_path}" ]; then
            echo "[$(date '+%F %T')] Missing model path: ${model_path}" >&2
            exit 1
        fi

        local embs="./item_info/${dataset}/${save_info}_${extraction_method}_item_embs.npy"
        if [ ! -f "${embs}" ]; then
            CUDA_VISIBLE_DEVICES="${cuda_device}" "${python_bin}" extract_llm_embedding.py \
                --dataset="${dataset}" \
                --model_path="${model_path}" \
                --item_prompt_type="${extraction_method}" \
                --bidirectional=1 \
                --save_info="${save_info}"
        else
            echo "[$(date '+%F %T')] Reuse embeddings: ${embs}"
        fi

        local port=$((12000 + RANDOM % 1000))
        CUDA_VISIBLE_DEVICES="${cuda_device}" "${python_bin}" -m accelerate.commands.launch \
            --main_process_port="${port}" repeated_evaluate_with_seqrec.py \
            --model=SASRec \
            --dataset="${dataset}" \
            --lr=1.0e-3 \
            --weight_decay=1.0e-4 \
            --embedding="${embs}" \
            --dropout=0.3 \
            --loss_type=ce \
            --run_id=CFKD

        echo "[$(date '+%F %T')] Finished dataset=${dataset}, model=${save_info}, gpu=${cuda_device}"
    done
}

pids=()
for i in "${!dataset_array[@]}"; do
    gpu="${gpu_array[$((i % ${#gpu_array[@]}))]}"
    dataset="${dataset_array[$i]}"
    if [ "${#gpu_array[@]}" -eq 1 ]; then
        run_dataset "${dataset}" "${gpu}" 2>&1 | tee -a "${log_dir}/${dataset}.log"
    else
        run_dataset "${dataset}" "${gpu}" > "${log_dir}/${dataset}.log" 2>&1 &
        pids+=("$!")
    fi
done

if [ "${#pids[@]}" -gt 0 ]; then
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
fi

echo "[$(date '+%F %T')] CFKD evaluation complete. Logs: ${log_dir}"
