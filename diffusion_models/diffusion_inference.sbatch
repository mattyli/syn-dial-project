#!/bin/bash
#SBATCH -A aip-zhu2048
#SBATCH -c 12
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"

export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

mkdir -p "$SLURM_TMPDIR/data"

rsync -a ~/projects/aip-zhu2048/mattli/syn-dial-project/aci-bench-main/data/challenge_data_json/ \
    "$SLURM_TMPDIR/data/"

python3 run_diffusion_downstream.py \
    --data_dir "$SLURM_TMPDIR/data" \
    --output_dir "/project/aip-zhu2048/mattli/syn-dial-project/diffusion_models/results"


