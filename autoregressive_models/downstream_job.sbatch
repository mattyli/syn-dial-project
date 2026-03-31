#!/bin/bash
#SBATCH --job-name=Meta-Llama-3.1-8B-Instruct-downstream
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=$HOME/.vec-inf-logs/Meta-Llama-3.1-8B-Instruct-downstream.%j.out
#SBATCH --error=$HOME/.vec-inf-logs/Meta-Llama-3.1-8B-Instruct-downstream.%j.err

# Activate your environment
# TODO: update this path to match your venv location
source $HOME/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate

# make directories on the compute node
mkdir -p "$SLURM_TMPDIR/data"
mkdir -p "$SLURM_TMPDIR/output"

# copy the aci-bench dataset over to the compute node (should this just be the entire aci-bench folder?)
rsync -a ~/projects/aip-zhu2048/mattli/syn-dial-project/aci-bench-main/data/challenge_data_json/ "$SLURM_TMPDIR/data/"

# Wait for the server to be ready using the job ID passed as CLI arg
python3 run_downstream.py "$SERVER_JOB_ID" "$MODEL_NAME"

sleep 30

# free resources (cancel the job)
scancel --user mattli
