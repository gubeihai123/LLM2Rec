#!/usr/bin/env bash

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-20000}"
DATASETS="${DATASETS:-Games_5core Goodreads}"
LOG_ROOT="${LOG_ROOT:-logs/continue_after_csft_$(date +%Y%m%d_%H%M%S)}"
CSFT_OUTPUT_DIR="./output/csft/Qwen2-0.5B-AmazonMix-6"

mkdir -p "${LOG_ROOT}"

free_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v min_free="${GPU_MIN_FREE_MB}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 >= min_free) print $1}'
}

echo "[$(date '+%F %T')] Simple continuation watcher started."

while true; do
    if [ -f "${CSFT_OUTPUT_DIR}/config.json" ] && ls "${CSFT_OUTPUT_DIR}"/model* >/dev/null 2>&1 && ! pgrep -f 'run_csft.py' >/dev/null; then
        echo "[$(date '+%F %T')] CSFT finished and final model exists."
        break
    fi

    if ! pgrep -f 'run_csft.py' >/dev/null && { [ ! -f "${CSFT_OUTPUT_DIR}/config.json" ] || ! ls "${CSFT_OUTPUT_DIR}"/model* >/dev/null 2>&1; }; then
        echo "[$(date '+%F %T')] CSFT stopped but final model files are missing in ${CSFT_OUTPUT_DIR}."
        exit 1
    fi

    echo "[$(date '+%F %T')] Waiting for CSFT. Sleep ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
done

while true; do
    mapfile -t gpus < <(free_gpus)
    if [ "${#gpus[@]}" -ge 1 ]; then
        break
    fi
    echo "[$(date '+%F %T')] Waiting for one free GPU for IEM."
    sleep "${POLL_SECONDS}"
done

if [ ! -d ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-1000 ]; then
    echo "[$(date '+%F %T')] Start IEM on GPU ${gpus[0]}."
    GPU_ID="${gpus[0]}" bash run_LLM2Rec_IEM.sh > "${LOG_ROOT}/02_iem.log" 2>&1
fi

dataset_count=$(wc -w <<< "${DATASETS}")
while true; do
    mapfile -t gpus < <(free_gpus)
    if [ "${#gpus[@]}" -ge "${dataset_count}" ]; then
        break
    fi
    echo "[$(date '+%F %T')] Waiting for ${dataset_count} free GPU(s) for evaluation."
    sleep "${POLL_SECONDS}"
done

eval_gpu_ids=$(IFS=,; echo "${gpus[*]:0:${dataset_count}}")
echo "[$(date '+%F %T')] Start evaluation on GPU(s): ${eval_gpu_ids}."
DATASETS="${DATASETS}" GPU_IDS="${eval_gpu_ids}" LOG_DIR="${LOG_ROOT}/03_eval" bash run_selected_extract_and_evaluate.sh > "${LOG_ROOT}/03_eval.log" 2>&1

echo "[$(date '+%F %T')] Simple continuation watcher complete."
