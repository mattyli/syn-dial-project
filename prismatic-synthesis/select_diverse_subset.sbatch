#!/bin/bash
#SBATCH -A aip-zhu2048
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca
#SBATCH --job-name=medsynth-subset-selection
#SBATCH --output=/project/aip-zhu2048/mattli/syn-dial-project/prismatic-synthesis/phase_1_run/medsynth-subset-selection/%x.%j.out
#SBATCH --error=/project/aip-zhu2048/mattli/syn-dial-project/prismatic-synthesis/phase_1_run/medsynth-subset-selection/%x.%j.err

# Runs gradient-based diverse subset selection on MedSynth.
# Env vars (set via --export or run_pipeline.sh):
#   SUBSET_SIZE   — number of examples to select (default: 1000)
#   K_CLUSTERS    — number of K-means clusters (default: n // 10)

module load cuda

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

PROJECT_ROOT="/project/aip-zhu2048/mattli/syn-dial-project"
DATASET_FILENAME="$PROJECT_ROOT/prismatic-synthesis/g-vendi/data/datasets/medsynth_selection.jsonl"
GRADIENT_DIR="$PROJECT_ROOT/prismatic-synthesis/g-vendi/data/gradient_storage/medsynth_selection--qwen3-0.6b"
OUTPUT_DIR="$PROJECT_ROOT/prismatic-synthesis/results"
SUBSET_SIZE="${SUBSET_SIZE:-1000}"

echo "Selecting $SUBSET_SIZE examples"
echo "Gradient dir: $GRADIENT_DIR"
echo "Output dir:   $OUTPUT_DIR"
[ -n "$K_CLUSTERS" ] && echo "K clusters:   $K_CLUSTERS"

cd "$PROJECT_ROOT"

K_ARG=""
[ -n "$K_CLUSTERS" ] && K_ARG="--k_clusters $K_CLUSTERS"

python prismatic-synthesis/select_diverse_subset.py \
    --dataset_filename "$DATASET_FILENAME" \
    --gradient_dir "$GRADIENT_DIR" \
    --subset_size "$SUBSET_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    $K_ARG
