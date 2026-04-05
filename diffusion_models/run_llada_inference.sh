#!/bin/bash

set -euo pipefail

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"

# set the HF Cache (in case its needed downstream)
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

SCRIPT_DIR="/project/aip-zhu2048/mattli/syn-dial-project"
mkdir -p "$SCRIPT_DIR/diffusion_models/results"

# submit the slurm job
sbatch \
    --chdir="$SCRIPT_DIR/diffusion_models" \
    --job-name="LLaDA2.1-mini-downstream" \
    --output="/project/6101844/mattli/syn-dial-project/diffusion_models/results/%x.%j.out" \
    --error="/project/6101844/mattli/syn-dial-project/diffusion_models/results/%x.%j.err" \
    diffusion_inference.sbatch
