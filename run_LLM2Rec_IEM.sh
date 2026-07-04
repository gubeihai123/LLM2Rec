# Second stage of training LLM2Rec -- Item Embedding Modeling.

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

model_path="${MODEL_PATH:-./models/Qwen2-0.5B}"
iem_stage1_dir="./output/iem_stage1/Qwen2-0.5B-AmazonMix6-CSFT"
gpu_id="${GPU_ID:-0}"
skip_mntp="${SKIP_MNTP:-1}"
mntp_config="${MNTP_CONFIG:-./llm2rec/train_mntp_config.json}"
simcse_config="${SIMCSE_CONFIG:-./llm2rec/train_simcse_config.json}"

if [ "${skip_mntp}" = "1" ]; then
    echo "Skipping Stage 2 - Train MNTP (SKIP_MNTP=1)"
else
    echo "Starting Stage 2 - Train MNTP..."
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=29501 ./llm2rec/run_mntp.py "${mntp_config}"
fi

# Stage 3 - Train SimCSE
echo "Starting Stage 3 - Train SimCSE..."
latest_mntp_ckpt=$(find "${iem_stage1_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [ -z "${latest_mntp_ckpt}" ] || [ ! -d "${latest_mntp_ckpt}" ]; then
    echo "No checkpoint found under ${iem_stage1_dir}; cannot start SimCSE." >&2
    exit 1
fi
if ls "${model_path}"/*token* >/dev/null 2>&1; then
    cp "${model_path}"/*token* "${latest_mntp_ckpt}/"
fi
CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=29502 ./llm2rec/run_unsupervised_SimCSE.py "${simcse_config}" "$@"
