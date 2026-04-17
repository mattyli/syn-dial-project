#!/bin/bash
# Run the full MedSynth diversity-selection pipeline.
#
# Usage (from project root):
#   bash prismatic-synthesis/run_pipeline.sh [SUBSET_SIZE]
#
# Steps:
#   1. Prepare data  — runs locally (fast, CPU-only)
#   2. Collect gradients — Slurm array job (8 x l40s GPU)
#   3. Select subset — Slurm job dependent on step 2 (1 x l40s GPU)

set -euo pipefail

SUBSET_SIZE="${1:-1000}"
PROJECT_ROOT="/project/aip-zhu2048/mattli/syn-dial-project"
LOG_ROOT="$PROJECT_ROOT/prismatic-synthesis/phase_1_run"

echo "=== MedSynth diversity-selection pipeline ==="
echo "Subset size: $SUBSET_SIZE"
echo "Project root: $PROJECT_ROOT"
echo

# ── Create log directories (Slurm requires them to exist before job starts) ───
mkdir -p "$LOG_ROOT/gvendi-medsynth-selection"
mkdir -p "$LOG_ROOT/medsynth-subset-selection"
echo "Log directory: $LOG_ROOT"
echo

# ── Step 1: Data preparation (local) ─────────────────────────────────────────
echo "[1/3] Preparing MedSynth JSONL data..."
source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

python "$PROJECT_ROOT/prismatic-synthesis/prepare_medsynth_data.py"

echo "Data preparation complete."
echo

# ── Step 2: Gradient collection (Slurm array, 8 GPUs) ────────────────────────
echo "[2/3] Submitting gradient collection array job..."
SBATCH_GRAD="$PROJECT_ROOT/prismatic-synthesis/g-vendi/scripts/collect_gradients_medsynth_selection.sbatch"
GRAD_JOB_ID=$(sbatch --parsable "$SBATCH_GRAD")
echo "  Gradient job ID: $GRAD_JOB_ID (array 0-7)"

# ── Step 3: Subset selection (Slurm, depends on all gradient tasks) ───────────
echo "[3/3] Submitting subset selection job (after $GRAD_JOB_ID)..."
SBATCH_SEL="$PROJECT_ROOT/prismatic-synthesis/select_diverse_subset.sbatch"
SEL_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"$GRAD_JOB_ID" \
    --export=ALL,SUBSET_SIZE="$SUBSET_SIZE" \
    "$SBATCH_SEL")
echo "  Selection job ID: $SEL_JOB_ID"

echo
echo "=== Jobs submitted ==="
echo "  Gradient collection : $GRAD_JOB_ID"
echo "  Subset selection    : $SEL_JOB_ID (runs after all gradient tasks finish)"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "Results will appear in:"
echo "  $PROJECT_ROOT/prismatic-synthesis/results/selected_subset_N${SUBSET_SIZE}.jsonl"
