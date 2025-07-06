#!/usr/bin/env bash
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export HF_ENDPOINT=https://hf-mirror.com

HF_MODEL="/absolute/path/to/LooK-Back/MLLMs/Solution-back-7B"
EVAL_DIR="/absolute/path/to/LooK-Back/eval"
DATA_DIR="$EVAL_DIR/eval_data"
RESULTS_DIR="${HF_MODEL%/actor/huggingface}/results"
mkdir -p "$RESULTS_DIR"

SYSTEM_PROMPT="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags, and use <back> </back> to verify your reasoning solution against the image, and rethink it in the <think> </think> tags based on your thinking and verification content. The final answer MUST BE put in \boxed{}, respectively, i.e., <think> reasoning process here </think> <back> verification process against the image here </back> <think> based on the thinking  and verification contents, rethinking here </think> \\boxed{final answer}."
cd "$EVAL_DIR"

python main.py \
  --model "$HF_MODEL" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets wemath,mathverse,mathvision,mathvista,GeoMath,hallubench,Super_Clevr,Tallyqa,MME \
  --tensor-parallel-size 4 \
  --system-prompt="$SYSTEM_PROMPT" \
  --min-pixels 262144 \
  --max-pixels 1000000 \
  --max-model-len 8192 \
  --temperature 0.0 \
  --version="back"


# python cal_score.py \
# --folder_path model_path/results