#!/usr/bin/env bash
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export HF_ENDPOINT=https://hf-mirror.com

HF_MODEL="/home/bml_job/custom_workspace/job-gguryn8pkf7k/zhurui/Look-Back/MLLMs/Semantic-back-7B"
EVAL_DIR="/home/bml_job/custom_workspace/job-gguryn8pkf7k/zhurui/Look-Back/eval"
DATA_DIR="$EVAL_DIR/eval_data"
RESULTS_DIR="${HF_MODEL%/actor/huggingface}/results"
mkdir -p "$RESULTS_DIR"

SYSTEM_PROMPT="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags, and use <back> </back> to verify your reasoning against the image. The final answer MUST BE put in \boxed{}, respectively, i.e., <think> reasoning process here </think> <back> verification process here </back> <think> continue reasoning </think> \\boxed{final answer}."
cd "$EVAL_DIR"

python main.py \
  --model "$HF_MODEL" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets wemath,mathverse,mathvision,GeoMath,mathvista,hallubench,Super_Clevr,Tallyqa,MME \
  --tensor-parallel-size 4 \
  --system-prompt="$SYSTEM_PROMPT" \
  --min-pixels 262144 \
  --max-pixels 1000000 \
  --max-model-len 8192 \
  --temperature 0.0 \
  --version="back"