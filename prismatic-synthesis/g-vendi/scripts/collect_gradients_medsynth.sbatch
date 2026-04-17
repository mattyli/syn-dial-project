#!/bin/bash
#SBATCH -A aip-zhu2048
#SBATCH --array=0-7
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca
#SBATCH --job-name=gvendi-medsynth

# Usage:
#   DATASET=notes     sbatch collect_gradients_medsynth.sbatch
#   DATASET=dialogues sbatch collect_gradients_medsynth.sbatch
#
# Submits 8 single-GPU array tasks; each processes 1/8 of the dataset.

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"
export BASE_MODELS_DIR="/model-weights"

MODEL_NAME="Qwen3-0.6B"
NUM_GPUS=8
SCRIPT_DIR="/project/aip-zhu2048/mattli/syn-dial-project/prismatic-synthesis/g-vendi"
DATASET="${DATASET:-notes}"
DATASET_FILENAME="$SCRIPT_DIR/data/datasets/medsynth_${DATASET}.jsonl"

echo "Task $SLURM_ARRAY_TASK_ID / $NUM_GPUS — dataset: $DATASET_FILENAME"

cd "$SCRIPT_DIR"

python collect_gradients.py \
    --model_name_or_path="$MODEL_NAME" \
    --dataset_filename="$DATASET_FILENAME" \
    --device_split_size=$NUM_GPUS \
    --device_idx=$SLURM_ARRAY_TASK_ID
