#!/bin/bash
# Slurm job: compute majority-voted pairwise preference scores for all 12 experiment pairs.
# Reads per-judge relative CSVs already in eval/benchmarking/results/<run_a>/ and writes
# relative_majority CSVs alongside them.  CPU-only, no GPU needed.
#
# Prerequisites: all 36 pairwise judge jobs from run_pairwise_all.sh must be complete.
#
# Submit:
#   cd eval/benchmarking
#   sbatch run_majority_vote_relative.sbatch

#SBATCH -A aip-zhu2048
#SBATCH --job-name=majority-vote-relative
#SBATCH --output=/project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking/results/majority_vote_relative_%j.out
#SBATCH --error=/project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking/results/majority_vote_relative_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca

set -euo pipefail

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

cd /project/aip-zhu2048/mattli/syn-dial-project/eval/benchmarking

echo "=== majority vote relative job start: $(date) ==="
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="

# Process all 12 pairs in one call (no --run_a = runs all canonical pairs)
python compute_majority_vote_relative.py

echo ""
echo "=== majority vote relative job end: $(date) ==="
