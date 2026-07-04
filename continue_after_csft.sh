#!/usr/bin/env bash
set -euo pipefail

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

poll_seconds="${POLL_SECONDS:-300}"
min_free_mb="${GPU_MIN_FREE_MB:-20000}"
selected_datasets="${DATASETS:-Games_5core Goodreads}"
log_root="${LOG_ROOT:-logs/continue_after_csft_$(date +%Y%m%d_%H%M%S)}"
csft_output_dir="./output/csft/Qwen2-0.5B-AmazonMix-6"
csft_pattern="./llm2rec/run_csft.py --base_model ./models/Qwen2-0.5B"

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
        echo "[$(date '+%F %T')] Need ${needed} GPU(s) with >= ${min_free_mb} MiB free; found ${#gpus[@]}. Sleep ${poll_seconds}s."
        nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits
        sleep "${poll_seconds}"
    done
}

echo "[$(date '+%F %T')] Continuation watcher started. Waiting for CSFT final model: ${csft_output_dir}"

while true; do
    if [ -f "${csft_output_dir}/config.json" ] && ls "${csft_output_dir}"/model* >/dev/null 2>&1 && ! pgrep -f "${csft_pattern}" >/dev/null; then
        break
    fi

    if { [ ! -f "${csft_output_dir}/config.json" ] || ! ls "${csft_output_dir}"/model* >/dev/null 2>&1; } && ! pgrep -f "${csft_pattern}" >/dev/null; then
        echo "[$(date '+%F %T')] CSFT is not running and final model is missing. Stop watcher."
        exit 1
    fi

    echo "[$(date '+%F %T')] CSFT still running or final model not ready. Sleep ${poll_seconds}s."
    sleep "${poll_seconds}"
done

if [ ! -d ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-1000 ]; then
    mapfile -t iem_gpus < <(wait_for_gpus 1)
    iem_gpu="${iem_gpus[0]}"
    echo "[$(date '+%F %T')] Start IEM on GPU: ${iem_gpu}"
    GPU_ID="${iem_gpu}" bash run_LLM2Rec_IEM.sh > "${log_root}/02_iem.log" 2>&1
else
    echo "[$(date '+%F %T')] IEM checkpoint exists; skip IEM."
fi

dataset_count=$(wc -w <<< "${selected_datasets}")
mapfile -t eval_gpus < <(wait_for_gpus "${dataset_count}")
eval_gpu_ids=$(IFS=,; echo "${eval_gpus[*]}")
echo "[$(date '+%F %T')] Start selected evaluation on GPU(s): ${eval_gpu_ids}"
DATASETS="${selected_datasets}" GPU_IDS="${eval_gpu_ids}" LOG_DIR="${log_root}/03_eval" bash run_selected_extract_and_evaluate.sh > "${log_root}/03_eval.log" 2>&1

echo "[$(date '+%F %T')] Continuation complete. Logs: ${log_root}"
