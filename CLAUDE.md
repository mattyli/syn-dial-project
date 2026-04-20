# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic dialogue generation project (targeting EMNLP 2026) focused on **diversity and data selection** within the MedSynth dataset. Diffusion models are no longer in scope — the project is exclusively AR-based.

The core pipeline:
1. **Note → Dialogue** (generation): Given a MedSynth clinical note, generate a realistic doctor-patient dialogue (few-shot with `MedSynth_huggingface_final.csv`).
2. **Dialogue → Note** (finetuning/evaluation): Finetune an AR model on MedSynth Dialogue→Note pairs, then benchmark on **ACI-bench** (`clinicalnlp_taskB_test1_full`).
3. **Diversity analysis** (`prismatic-synthesis/`): G-Vendi clustering to study and select diverse subsets of MedSynth.

Models in use:
- **AR**: Meta-Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, etc. — served via `vec-inf` (vLLM)

## Environment Setup

Activate the project venv before running any scripts:
```bash
source ~/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"
```

The project directory is symlinked: `/project/6101844/mattli/syn-dial-project` ↔ `/project/aip-zhu2048/mattli/syn-dial-project`. Slurm scripts use the `aip-zhu2048` path.

## Commands

**AR inference (via vec-inf + Slurm):**
```bash
# Launches vec-inf server job, then submits downstream inference as a dependent job
cd autoregressive_models
bash run_autoregressive_inference.sh Meta-Llama-3.1-8B-Instruct
# No arg = launches all four target models in parallel:
#   medgemma-27b-text-it, gemma-4-26B-A4B-it, Qwen3.5-27B, aya-expanse-32b
# Allowed models: Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
#   Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B, gpt-oss-20b,
#   medgemma-27b-text-it, gemma-4-26B-A4B-it, Qwen3.5-27B, aya-expanse-32b
```

**ACI-bench evaluation:**
```bash
cd aci-bench-main/evaluation
# Requires three args: gold CSV, system JSONL/CSV, metadata CSV
python3 evaluate_fullnote.py <gold_csv> <pred_file> [metadata_csv]
# Results written to aci-bench-main/evaluation/results/<pred_filename>.json
# Note: hardcodes CUDA_VISIBLE_DEVICES=1; run on a GPU node
```

## Architecture

### This repo (`syn-dial-project`)

```
autoregressive_models/
├── run_autoregressive_inference.sh  # Orchestrator: launches vec-inf server + downstream job
├── downstream_job.sbatch            # Slurm job: waits for server, runs run_downstream.py
└── run_downstream.py                # Calls OpenAI-compatible vec-inf API, writes JSONL results

MedSynth_huggingface_final.csv       # Source of few-shot Note/Dialogue examples
aci-bench-main/                      # ACI-bench dataset + evaluation scripts
prismatic-synthesis/                 # G-Vendi clustering/diversity work (separate pipeline)
```

**AR inference pipeline:** `run_autoregressive_inference.sh` calls `vec-inf launch <MODEL>` to get a SLURM job ID for the inference server, then submits `downstream_job.sbatch` with `--dependency=after:<job_id>`. The downstream job calls `run_downstream.py`, which uses `VecInfClient.wait_until_ready()` then hits the OpenAI-compatible endpoint. After inference, `scancel` frees the server job.

**Output format:** Inference writes JSONL with `{"file", "src", "tgt", "prediction"}` to `autoregressive_models/results/<MODEL_NAME>_clinicalnlp_taskB_test1_full.jsonl`.

## Code Guidelines

- Include a docstring at the top of each new Python file with how to run it.
- Read existing scripts before writing new ones.
- Use absolute paths in documentation and scripts.
