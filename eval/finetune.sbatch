#!/bin/bash
# Slurm job for QLoRA finetuning on an L40s.
# Do not submit this directly — use run_finetune.sh which sets --output/--error/--job-name
# and creates the results directory before submission.
#
# Required env vars (set by run_finetune.sh via --export):
#   DATA_PATH      - path to MedSynth_huggingface_final.csv
#   OUTPUT_DIR     - where to write the adapter + checkpoints
#   BASE_MODEL     - unsloth model name (default set below if unset)
#   NUM_SAMPLES    - number of training rows (empty = full dataset)
#   EPOCHS         - number of training epochs
#   LORA_R         - LoRA rank
#   SEED           - random seed

#SBATCH -A aip-zhu2048
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca

set -euo pipefail

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

cd /project/aip-zhu2048/mattli/syn-dial-project/eval

# activate the venv
source ../.venv/bin/activate
module load cuda
# Build the argument list; --num_samples is optional
ARGS=(
    --data_path  "${DATA_PATH:-/project/aip-zhu2048/mattli/syn-dial-project/MedSynth_huggingface_final.csv}"
    --output_dir "${OUTPUT_DIR}"
    --base_model "${BASE_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
    --epochs     "${EPOCHS:-4}"
    --lora_r     "${LORA_R:-16}"
    --seed       "${SEED:-42}"
)

if [[ -n "${NUM_SAMPLES:-}" ]]; then
    ARGS+=(--num_samples "$NUM_SAMPLES")
fi

# Compute Canada installs numpy with a local version tag (+computecanada) that causes
# unsloth_zoo to detect a version mismatch at import time. Reinstall to strip the tag.
# uv pip install --quiet --force-reinstall --no-deps "numpy==1.26.4"

echo "=== finetune job start: $(date) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== args: ${ARGS[*]} ==="

python finetune.py "${ARGS[@]}"

echo "=== finetune job end: $(date) ==="
