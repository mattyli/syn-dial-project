#!/bin/bash
#SBATCH -A aip-zhu2048
#SBATCH -c 12
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.li@queensu.ca

# Activate your environment
# TODO: update this path to match your venv location
source "$HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate"

# make directories on the compute node
mkdir -p "$SLURM_TMPDIR/data"
mkdir -p "$SLURM_TMPDIR/output"

# copy the aci-bench dataset over to the compute node (should this just be the entire aci-bench folder?)
rsync -a ~/projects/aip-zhu2048/mattli/syn-dial-project/aci-bench-main/data/challenge_data_json/ "$SLURM_TMPDIR/data/"

# ERR here (can't resolve this path after path was changed in the upstream sbatch command
# Wait for the server to be ready using the job ID passed as CLI arg
python3 run_downstream.py --job_id "$SERVER_JOB_ID" --model_name "$MODEL_NAME" \
    --data_dir "$SLURM_TMPDIR/data" \
    --output_dir "/project/aip-zhu2048/mattli/syn-dial-project/autoregressive_models/results"

sleep 30

# free resources (cancel only this model's vec-inf server job)
scancel "$SERVER_JOB_ID"
