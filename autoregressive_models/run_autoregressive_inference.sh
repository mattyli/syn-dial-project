#!/bin/bash
set -euo pipefail

# Script that runs inference for AR models on the train/valid + test1 sets of ACI bench

# Allowed models
ALLOWED_MODELS=(
  "Meta-Llama-3.1-8B-Instruct"
  "Meta-Llama-3.1-70B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "Qwen2.5-7B-Instruct"
)

# Default model, or first CLI arg if provided
MODEL_NAME="${1:-Meta-Llama-3.1-8B-Instruct}"

# Validate against allowed values
case "$MODEL_NAME" in
  "Meta-Llama-3.1-8B-Instruct" \
  |"Meta-Llama-3.1-70B-Instruct" \
  |"Mistral-7B-Instruct-v0.3" \
  |"Qwen2.5-7B-Instruct"\
  |"gpt-oss-20b"\
  |"Qwen3-8B"
	)
    ;;
  *)
    echo "Error: invalid model '$MODEL_NAME'"
    echo "Allowed models:"
    printf '  - %s\n' "${ALLOWED_MODELS[@]}"
    exit 1
    ;;
esac

LAUNCH_ARGS="$MODEL_NAME"

# --- Directly from https://github.com/VectorInstitute/vector-inference/blob/main/examples/slurm_dependency/run_workflow.sh

# ---- Step 1: Launch the server
RAW_JSON=$(vec-inf launch "$LAUNCH_ARGS" --json-mode)
SERVER_JOB_ID=$(echo "$RAW_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['slurm_job_id'])")
echo "Launched server as job $SERVER_JOB_ID"
echo "$RAW_JSON"

# ---- Step 2: Submit downstream job
sbatch --dependency=after:$SERVER_JOB_ID \
  --export=SERVER_JOB_ID="$SERVER_JOB_ID",MODEL_NAME="$MODEL_NAME" \
  downstream_job.sbatch
