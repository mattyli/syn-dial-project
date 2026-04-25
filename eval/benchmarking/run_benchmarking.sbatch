#!/bin/bash
# Slurm job for benchmarking a finetuned Dialogue→Note model on ACI-bench.
# Do not submit this directly — use run_benchmarking.sh which sets
# --output/--error/--job-name and creates the results directory.
#
# Required env vars (set by run_benchmarking.sh via --export):
#   RUN_NAME      - name matching eval/results/<RUN_NAME>/model/final_model/
#
# Optional env vars:
#   RUN_NAME_B    - second run for pairwise relative jury (default: unset)
#   JUDGE         - prometheus or gpt (default: prometheus)
#   SKIP_AUTO     - set to 1 to skip traditional metrics
#   SKIP_LLM      - set to 1 to skip LLM jury

#SBATCH -A aip-zhu2048
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca

set -euo pipefail

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

module load cuda
# activate venv
source ../../.venv/bin/activate

# Fix numpy local version tag that breaks unsloth_zoo import on Compute Canada
#uv pip install --quiet --force-reinstall --no-deps "numpy==1.26.4"

cd /project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking

ARGS=(--run-name "${RUN_NAME}")

if [[ -n "${RUN_NAME_B:-}" ]]; then
    ARGS+=(--run-name-b "$RUN_NAME_B")
fi

if [[ -n "${JUDGE:-}" ]]; then
    ARGS+=(--judge "$JUDGE")
fi

if [[ "${SKIP_AUTO:-0}" == "1" ]]; then
    ARGS+=(--skip-auto-metrics)
fi

if [[ "${SKIP_LLM:-0}" == "1" ]]; then
    ARGS+=(--skip-llm-judge)
fi

echo "=== benchmarking job start: $(date) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== args: ${ARGS[*]} ==="

python run_benchmarking.py "${ARGS[@]}"

echo "=== benchmarking job end: $(date) ==="
