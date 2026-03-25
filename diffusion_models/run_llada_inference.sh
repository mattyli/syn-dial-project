#!/bin/bash

set -euo pipefail

source .venv/bin/activate

# set the HF Cache (incase its needed downstream)

export HF_HOME=~/projects/aip-zhu2048/mattli/hf_cache

# submit the slurm job
