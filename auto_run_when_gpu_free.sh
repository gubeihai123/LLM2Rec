#!/usr/bin/env bash
set -euo pipefail

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

poll_seconds="${POLL_SECONDS:-300}"
min_free_mb="${GPU_MIN_FREE_MB:-20000}"
selected_datasets="${DATASETS:-Games_5core Goodreads}"
csft_max_gpus="${CSFT_MAX_GPUS:-2}"
log_root="${LOG_ROOT:-logs/auto_run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${log_root}"

free_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v min_free="${min_free_mb}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 >= min_free) print $1}'
}

wait_for_gpus() {
    local needed="$1"
    while true; do
        mapfile -t gpus < <(free_gpus)
        if [ "${#gpus[@]}" -ge "${needed}" ]; then
            printf '%s\n' "${gpus[@]:0:${needed}}"
            return 0
        fi
        echo "[$(date '+%F %T')] Need ${needed} GPU(s) with >= ${min_free_mb} MiB free; found ${#gpus[@]}. Sleep ${poll_seconds}s." >&2
        nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits >&2
        sleep "${poll_seconds}"
    done
}

pick_gpus_up_to() {
    local max_count="$1"
    mapfile -t gpus < <(wait_for_gpus 1)
    mapfile -t gpus < <(free_gpus)
    local count="${#gpus[@]}"
    if [ "${count}" -gt "${max_count}" ]; then
        count="${max_count}"
    fi
    printf '%s\n' "${gpus[@]:0:${count}}"
}

echo "[$(date '+%F %T')] Auto runner started. min_free_mb=${min_free_mb}, datasets='${selected_datasets}'"

if [ ! -f ./output/csft/Qwen2-0.5B-AmazonMix-6/config.json ] || ! ls ./output/csft/Qwen2-0.5B-AmazonMix-6/model* >/dev/null 2>&1; then
    mapfile -t csft_gpus < <(pick_gpus_up_to "${csft_max_gpus}")
    csft_gpu_ids=$(IFS=,; echo "${csft_gpus[*]}")
    echo "[$(date '+%F %T')] Start CSFT on GPU(s): ${csft_gpu_ids}"
    GPU_IDS="${csft_gpu_ids}" bash run_LLM2Rec_CSFT.sh > "${log_root}/01_csft.log" 2>&1
else
    echo "[$(date '+%F %T')] CSFT final model exists; skip CSFT."
fi

if [ ! -d ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-1000 ]; then
    mapfile -t iem_gpus < <(wait_for_gpus 1)
    iem_gpu="${iem_gpus[0]}"
    echo "[$(date '+%F %T')] Start IEM on GPU: ${iem_gpu}"
    GPU_ID="${iem_gpu}" bash run_LLM2Rec_IEM.sh > "${log_root}/02_iem.log" 2>&1
else
    echo "[$(date '+%F %T')] IEM checkpoint exists; skip IEM."
fi

dataset_count=$(wc -w <<< "${selected_datasets}")
mapfile -t eval_gpus < <(pick_gpus_up_to "${dataset_count}")
eval_gpu_ids=$(IFS=,; echo "${eval_gpus[*]}")
echo "[$(date '+%F %T')] Start selected evaluation on GPU(s): ${eval_gpu_ids}"
DATASETS="${selected_datasets}" GPU_IDS="${eval_gpu_ids}" LOG_DIR="${log_root}/03_eval" bash run_selected_extract_and_evaluate.sh > "${log_root}/03_eval.log" 2>&1

echo "[$(date '+%F %T')] Auto runner complete. Logs: ${log_root}"
