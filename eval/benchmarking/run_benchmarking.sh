#!/usr/bin/env bash
# Launch a benchmarking Slurm job for a finetuned Dialogue→Note model.
#
# Usage:
#   bash run_benchmarking.sh --run_name medsynth-full [options]
#
# Options:
#   --run_name NAME     eval/results/<NAME>/model/final_model/ (required)
#   --run_name_b NAME   second run for pairwise relative jury
#   --judge TYPE        prometheus | gemma3_27b | qwen25_32b (default: prometheus)
#                       gemma3_27b = gemma-3-27b-it
#                       qwen25_32b = Qwen2.5-32B-Instruct
#                       Run all three, then use compute_majority_vote.py to aggregate
#   --skip_auto         skip traditional metrics
#   --skip_llm          skip LLM jury
#
# Outputs:
#   eval/benchmarking/results/<run_name>/slurm.out
#   eval/benchmarking/results/<run_name>/slurm.err
#   eval/benchmarking/results/<run_name>/auto_metrics.json
#   eval/benchmarking/results/<run_name>/<aspect>_Absolute_prometheus_scores_*.csv
#   eval/benchmarking/results/<run_name>/promethus_relative_score_*.csv  (if --run_name_b)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_NAME=""
RUN_NAME_B=""
JUDGE=""
SKIP_AUTO="0"
SKIP_LLM="0"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)   RUN_NAME="$2";   shift 2 ;;
        --run_name_b) RUN_NAME_B="$2"; shift 2 ;;
        --judge)      JUDGE="$2";      shift 2 ;;
        --skip_auto)  SKIP_AUTO="1";   shift ;;
        --skip_llm)   SKIP_LLM="1";    shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --run_name is required" >&2
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

JUDGE_LABEL="${JUDGE:-prometheus}"

echo "Submitting benchmarking job: $RUN_NAME (judge: $JUDGE_LABEL)"
echo "  Results dir : $RESULTS_DIR"
[[ -n "$RUN_NAME_B" ]] && echo "  Compare to  : $RUN_NAME_B"

sbatch \
    --job-name="bench-$RUN_NAME-$JUDGE_LABEL" \
    --output="$RESULTS_DIR/slurm_${JUDGE_LABEL}.out" \
    --error="$RESULTS_DIR/slurm_${JUDGE_LABEL}.err" \
    --export="ALL,RUN_NAME=$RUN_NAME,RUN_NAME_B=$RUN_NAME_B,JUDGE=$JUDGE,SKIP_AUTO=$SKIP_AUTO,SKIP_LLM=$SKIP_LLM" \
    "$SCRIPT_DIR/run_benchmarking.sbatch"
