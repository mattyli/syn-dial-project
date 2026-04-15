#!/bin/bash
set -euo pipefail

# Script that runs inference for AR models on the train/valid + test1 sets of ACI bench.
# Usage:
#   bash run_autoregressive_inference.sh                  # run all four target models sequentially
#   bash run_autoregressive_inference.sh <MODEL_NAME>     # run a single model
source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"

# Allowed models
ALLOWED_MODELS=(
  "Meta-Llama-3.1-8B-Instruct"
  "Meta-Llama-3.1-70B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "Qwen2.5-7B-Instruct"
  "Qwen3-8B"
  "gpt-oss-20b"
  "medgemma-27b-text-it"
  "gemma-4-26B-A4B-it"
  "Qwen3.5-27B"
  "aya-expanse-32b"
)

SCRIPT_DIR="/project/aip-zhu2048/mattli/syn-dial-project"
RESULTS_DIR="$SCRIPT_DIR/autoregressive_models/results"
mkdir -p "$RESULTS_DIR"

# ---- Core: launch vec-inf server + submit downstream job for one model ----
launch_model() {
  local model_name="$1"

  echo ""
  echo "=== Launching inference for $model_name ==="

  # Step 1: Launch the vec-inf server
  RAW_JSON=$(vec-inf launch "$model_name" --work-dir="$SCRIPT_DIR" --json-mode)
  SERVER_JOB_ID=$(echo "$RAW_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['slurm_job_id'])")
  echo "Server job: $SERVER_JOB_ID"
  echo "$RAW_JSON"

  # Step 2: Create model-specific results folder and submit downstream job
  MODEL_RESULTS_DIR="$RESULTS_DIR/$model_name"
  mkdir -p "$MODEL_RESULTS_DIR"

  DOWNSTREAM_JOB_ID=$(sbatch \
    --chdir="$SCRIPT_DIR/autoregressive_models" \
    --dependency=after:"$SERVER_JOB_ID" \
    --job-name="${model_name}-downstream" \
    --output="$MODEL_RESULTS_DIR/%x.%j.out" \
    --error="$MODEL_RESULTS_DIR/%x.%j.err" \
    --export=SERVER_JOB_ID="$SERVER_JOB_ID",MODEL_NAME="$model_name" \
    downstream_job.sbatch \
    | awk '{print $NF}')
  echo "Downstream job: $DOWNSTREAM_JOB_ID"
}

# ---- Validate a model name against the allowed list ----
validate_model() {
  local model_name="$1"
  case "$model_name" in
    "Meta-Llama-3.1-8B-Instruct" \
    |"Meta-Llama-3.1-70B-Instruct" \
    |"Mistral-7B-Instruct-v0.3" \
    |"Qwen2.5-7B-Instruct" \
    |"gpt-oss-20b" \
    |"Qwen3-8B" \
    |"medgemma-27b-text-it" \
    |"gemma-4-26B-A4B-it" \
    |"Qwen3.5-27B" \
    |"aya-expanse-32b" \
    )
      ;;
    *)
      echo "Error: invalid model '$model_name'"
      echo "Allowed models:"
      printf '  - %s\n' "${ALLOWED_MODELS[@]}"
      exit 1
      ;;
  esac
}

# ---- Entry point ----

if [[ $# -eq 0 ]]; then
  # No argument: launch all four target models in parallel.
  # Each downstream job cancels only its own vec-inf server job when done.
  LOOP_MODELS=(
    "medgemma-27b-text-it"
    "gemma-4-26B-A4B-it"
    "Qwen3.5-27B"
    "aya-expanse-32b"
  )
  for MODEL in "${LOOP_MODELS[@]}"; do
    launch_model "$MODEL"
  done
  echo ""
  echo "All jobs submitted."

else
  # Single-model mode
  MODEL_NAME="$1"
  validate_model "$MODEL_NAME"
  launch_model "$MODEL_NAME"
fi
