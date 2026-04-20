#!/usr/bin/env bash
# Launch a MedSynth finetuning Slurm job.
#
# Usage:
#   bash run_finetune.sh [options]
#
# Options:
#   --job_name NAME       Slurm job name and results subdirectory (default: medsynth-finetune)
#   --num_samples N       Training rows to use (default: full dataset)
#   --base_model NAME     unsloth model id (default: unsloth/Meta-Llama-3.1-8B-Instruct)
#   --epochs N            Training epochs (default: 4)
#   --lora_r N            LoRA rank (default: 16)
#   --seed N              Random seed (default: 42)
#   --data_path PATH      Path to MedSynth CSV (default: project root CSV)
#
# Results are written to:
#   eval/results/<job_name>/slurm.out   (stdout)
#   eval/results/<job_name>/slurm.err   (stderr)
#   eval/results/<job_name>/model/      (saved adapter)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Use the aip-zhu2048 path for Slurm (shared filesystem)
PROJECT_SLURM="/project/aip-zhu2048/mattli/syn-dial-project"

# ── Defaults ──────────────────────────────────────────────────────────────────
JOB_NAME="medsynth-finetune"
NUM_SAMPLES=""
BASE_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
EPOCHS="4"
LORA_R="16"
SEED="42"
DATA_PATH="$PROJECT_SLURM/MedSynth_huggingface_final.csv"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --job_name)   JOB_NAME="$2";   shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --base_model) BASE_MODEL="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --lora_r)     LORA_R="$2";     shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --data_path)  DATA_PATH="$2";  shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

RESULTS_DIR="$SCRIPT_DIR/results/$JOB_NAME"
OUTPUT_DIR="$PROJECT_SLURM/eval/results/$JOB_NAME/model"

mkdir -p "$RESULTS_DIR"

echo "Submitting finetuning job: $JOB_NAME"
echo "  Results dir : $RESULTS_DIR"
echo "  Model output: $OUTPUT_DIR"
echo "  num_samples : ${NUM_SAMPLES:-<full dataset>}"
echo "  base_model  : $BASE_MODEL"
echo "  epochs      : $EPOCHS"

sbatch \
    --job-name="$JOB_NAME" \
    --output="$RESULTS_DIR/slurm.out" \
    --error="$RESULTS_DIR/slurm.err" \
    --export="ALL,DATA_PATH=$DATA_PATH,OUTPUT_DIR=$OUTPUT_DIR,BASE_MODEL=$BASE_MODEL,NUM_SAMPLES=$NUM_SAMPLES,EPOCHS=$EPOCHS,LORA_R=$LORA_R,SEED=$SEED" \
    "$SCRIPT_DIR/finetune.sbatch"
