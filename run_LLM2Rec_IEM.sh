# Second stage of training LLM2Rec -- Item Embedding Modeling.

if [ -f ./setup_env.sh ]; then
    source ./setup_env.sh
fi

model_path="${MODEL_PATH:-./models/Qwen2-0.5B}"
iem_stage1_dir="./output/iem_stage1/Qwen2-0.5B-AmazonMix6-CSFT"
gpu_id="${GPU_ID:-0}"

# Stage 2 - Train MNTP
echo "Starting Stage 2 - Train MNTP..."
CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=29501 ./llm2rec/run_mntp.py ./llm2rec/train_mntp_config.json

# Stage 3 - Train SimCSE
echo "Starting Stage 3 - Train SimCSE..."
latest_mntp_ckpt=$(ls -d ${iem_stage1_dir}/checkpoint-* | sort -V | tail -n 1)
cp ${model_path}/*token* ${latest_mntp_ckpt}/
CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=29502 ./llm2rec/run_unsupervised_SimCSE.py ./llm2rec/train_simcse_config.json
