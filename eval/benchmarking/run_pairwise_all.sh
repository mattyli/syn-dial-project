#!/usr/bin/env bash
# Submit all 12 pairwise relative-jury jobs (3 judges × 12 pairs).
# Each job compares a diverse-subset model against the full-MedSynth model
# of the same architecture family.
#
# Predictions must already be cached (eval/benchmarking/results/<run>/predictions.jsonl)
# from prior absolute benchmarking runs — inference is skipped.
#
# Usage:
#   cd eval/benchmarking
#   bash run_pairwise_all.sh
#
# Outputs per job:
#   results/<run_a>/promethus_relative_score_<run_a>_<run_b>_<date>.csv
#   results/<run_a>/relative_gemma3_27b_<run_a>_<run_b>_<date>.csv
#   results/<run_a>/relative_qwen25_32b_<run_a>_<run_b>_<date>.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGES=(prometheus gemma3_27b qwen25_32b)

declare -A FULL_RUN=(
    [gemma3-4b]=finetune-full-medsynth-gemma3-4b-3ep
    [llama32-3b]=finetune-full-medsynth-llama32-3b-3ep
    [qwen25-3b]=finetune-full-medsynth-qwen25-3b-3ep
)

PAIRS=(
    "finetune-diverse-k100-n1000-gemma3-4b-3ep gemma3-4b"
    "finetune-diverse-k100-n2500-gemma3-4b-3ep gemma3-4b"
    "finetune-diverse-k100-n5000-gemma3-4b-3ep gemma3-4b"
    "finetune-diverse-k100-n7500-gemma3-4b-3ep gemma3-4b"
    "finetune-diverse-k100-n1000-llama32-3b-3ep llama32-3b"
    "finetune-diverse-k100-n2500-llama32-3b-3ep llama32-3b"
    "finetune-diverse-k100-n5000-llama32-3b-3ep llama32-3b"
    "finetune-diverse-k100-n7500-llama32-3b-3ep llama32-3b"
    "finetune-diverse-k100-n1000-qwen25-3b-3ep qwen25-3b"
    "finetune-diverse-k100-n2500-qwen25-3b-3ep qwen25-3b"
    "finetune-diverse-k100-n5000-qwen25-3b-3ep qwen25-3b"
    "finetune-diverse-k100-n7500-qwen25-3b-3ep qwen25-3b"
)

for pair in "${PAIRS[@]}"; do
    run_a="${pair%% *}"
    family="${pair##* }"
    run_b="${FULL_RUN[$family]}"
    for judge in "${JUDGES[@]}"; do
        echo "Submitting: $run_a vs $run_b  judge=$judge"
        bash "$SCRIPT_DIR/run_benchmarking.sh" \
            --run_name "$run_a" \
            --run_name_b "$run_b" \
            --judge "$judge" \
            --skip_auto
    done
done

echo ""
echo "Submitted $((${#PAIRS[@]} * ${#JUDGES[@]})) jobs total."
echo "After all jobs complete, run:"
echo "  sbatch run_majority_vote_relative.sbatch"
