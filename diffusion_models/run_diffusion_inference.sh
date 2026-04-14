#!/bin/bash
# Usage: bash run_diffusion_inference.sh [llada|dream]
#   llada  → LLaDA2.1-mini  (default)
#   dream  → Dream-v0-Instruct-7B

set -euo pipefail

MODEL=${1:-llada}
if [[ "$MODEL" != "llada" && "$MODEL" != "dream" ]]; then
    echo "Usage: $0 [llada|dream]" >&2
    exit 1
fi

source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

SCRIPT_DIR="/project/aip-zhu2048/mattli/syn-dial-project"
mkdir -p "$SCRIPT_DIR/diffusion_models/results"

sbatch \
    --export=ALL,MODEL="$MODEL" \
    --chdir="$SCRIPT_DIR/diffusion_models" \
    --job-name="diffusion-downstream-$MODEL" \
    --output="/project/6101844/mattli/syn-dial-project/diffusion_models/results/%x.%j.out" \
    --error="/project/6101844/mattli/syn-dial-project/diffusion_models/results/%x.%j.err" \
    diffusion_inference.sbatch
