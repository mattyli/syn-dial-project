#!/bin/bash
# Slurm job: compute majority-voted LLM judge scores for all 15 experiment runs.
# Reads per-judge CSVs already in eval/benchmarking/results/<run_name>/ and writes
# majority_scores CSVs alongside them.  CPU-only, no GPU needed.
#
# Submit:
#   cd eval/benchmarking
#   sbatch run_majority_vote.sbatch

#SBATCH -A aip-zhu2048
#SBATCH --job-name=majority-vote-all
#SBATCH --output=/project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking/results/majority_vote_%j.out
#SBATCH --error=/project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking/results/majority_vote_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca

set -euo pipefail

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

cd /project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking

RUNS=(
    finetune-diverse-k100-n1000-gemma3-4b-3ep
    finetune-diverse-k100-n1000-llama32-3b-3ep
    finetune-diverse-k100-n1000-qwen25-3b-3ep
    finetune-diverse-k100-n2500-gemma3-4b-3ep
    finetune-diverse-k100-n2500-llama32-3b-3ep
    finetune-diverse-k100-n2500-qwen25-3b-3ep
    finetune-diverse-k100-n5000-gemma3-4b-3ep
    finetune-diverse-k100-n5000-llama32-3b-3ep
    finetune-diverse-k100-n5000-qwen25-3b-3ep
    finetune-diverse-k100-n7500-gemma3-4b-3ep
    finetune-diverse-k100-n7500-llama32-3b-3ep
    finetune-diverse-k100-n7500-qwen25-3b-3ep
    finetune-full-medsynth-gemma3-4b-3ep
    finetune-full-medsynth-llama32-3b-3ep
    finetune-full-medsynth-qwen25-3b-3ep
)

echo "=== majority vote job start: $(date) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="

for run in "${RUNS[@]}"; do
    echo ""
    echo "--- $run ---"
    python compute_majority_vote.py --run_name "$run"
done

echo ""
echo "=== majority vote job end: $(date) ==="
