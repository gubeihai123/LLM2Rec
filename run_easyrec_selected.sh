#!/usr/bin/env bash
set -euo pipefail

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

datasets="${DATASETS:-Games_5core Arts_5core Movies_5core Sports_5core Baby_5core Goodreads}"
gpu_ids="${GPU_IDS:-0}"
log_dir="${LOG_DIR:-logs/easyrec_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${log_dir}"

IFS=' ' read -r -a dataset_array <<< "${datasets}"
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"

run_dataset() {
    local dataset="$1"
    local cuda_device="$2"
    local emb="./item_info/${dataset}/EasyRec_title_item_embs.npy"
    local port=$((14000 + RANDOM % 1000))

    echo "[$(date '+%F %T')] Start EasyRec dataset=${dataset}, gpu=${cuda_device}"
    if [ ! -f "${emb}" ]; then
        CUDA_VISIBLE_DEVICES="${cuda_device}" python -c "from Baseline_inference import main; main(model_name='easyrec', mode='item', dataset_name='${dataset}')"
    else
        echo "[$(date '+%F %T')] Reuse existing embedding: ${emb}"
    fi

    CUDA_VISIBLE_DEVICES="${cuda_device}" accelerate launch --main_process_port="${port}" repeated_evaluate_with_seqrec.py \
        --model=SASRec \
        --dataset="${dataset}" \
        --lr=1.0e-3 \
        --weight_decay=1.0e-4 \
        --embedding="${emb}" \
        --dropout=0.3 \
        --loss_type=ce \
        --run_id=EasyRec

    echo "[$(date '+%F %T')] Finished EasyRec dataset=${dataset}, gpu=${cuda_device}"
}

worker() {
    local worker_index="$1"
    local cuda_device="$2"

    for i in "${!dataset_array[@]}"; do
        if [ "$((i % ${#gpu_array[@]}))" -eq "${worker_index}" ]; then
            dataset="${dataset_array[$i]}"
            run_dataset "${dataset}" "${cuda_device}" 2>&1 | tee -a "${log_dir}/${dataset}.log"
        fi
    done
}

pids=()
for i in "${!gpu_array[@]}"; do
    if [ "${#gpu_array[@]}" -eq 1 ]; then
        worker "${i}" "${gpu_array[$i]}"
    else
        worker "${i}" "${gpu_array[$i]}" > "${log_dir}/gpu_${gpu_array[$i]}.log" 2>&1 &
        pids+=("$!")
    fi
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done

echo "[$(date '+%F %T')] EasyRec complete. Logs: ${log_dir}"
