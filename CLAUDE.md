# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic dialogue generation project (targeting EMNLP 2026) comparing AR and diffusion LMs on clinical note generation. The downstream evaluation task is **ACI-bench** (doctor-patient dialogue → structured clinical note).

Models in use:
- **AR**: Meta-Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, etc. — served via `vec-inf` (vLLM)
- **Diffusion**: LLaDA2.1-mini — loaded directly via the `dllm` library

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
# Allowed models: Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
#   Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B, gpt-oss-20b
```

**Diffusion inference (LLaDA2.1-mini via Slurm):**
```bash
cd diffusion_models
bash run_llada_inference.sh
# Or directly (requires GPU):
python3 run_diffusion_downstream.py --data_dir /path/to/data --output_dir ./results
```

**ACI-bench evaluation:**
```bash
cd aci-bench-main/evaluation
# Requires three args: gold CSV, system JSONL/CSV, metadata CSV
python3 evaluate_fullnote.py <gold_csv> <pred_file> [metadata_csv]
# Results written to aci-bench-main/evaluation/results/<pred_filename>.json
# Note: hardcodes CUDA_VISIBLE_DEVICES=1; run on a GPU node
```

**dllm tests:**
```bash
cd /project/6101844/mattli/dllm && pytest scripts/tests -v -ra
```

## Architecture

### This repo (`syn-dial-project`)

```
autoregressive_models/
├── run_autoregressive_inference.sh  # Orchestrator: launches vec-inf server + downstream job
├── downstream_job.sbatch            # Slurm job: waits for server, runs run_downstream.py
└── run_downstream.py                # Calls OpenAI-compatible vec-inf API, writes JSONL results

diffusion_models/
├── run_llada_inference.sh           # Slurm submission wrapper
├── diffusion_inference.sbatch       # Slurm job spec (l40s GPU, 1hr)
└── run_diffusion_downstream.py      # ACI-bench inference loop, writes JSONL results

aci-bench-main/                      # ACI-bench dataset + evaluation scripts (submodule/copy)
```

**AR inference pipeline:** `run_autoregressive_inference.sh` calls `vec-inf launch <MODEL>` to get a SLURM job ID for the inference server, then submits `downstream_job.sbatch` with `--dependency=after:<job_id>`. The downstream job calls `run_downstream.py`, which uses `VecInfClient.wait_until_ready()` then hits the OpenAI-compatible endpoint.

**Diffusion inference pipeline:** `run_diffusion_downstream.py` loads LLaDA2.1-mini via `AutoModelForCausalLM.from_pretrained` (direct HuggingFace, `trust_remote_code=True`), calls `model.generate()` with diffusion-specific args (`gen_length`, `block_length`, `threshold`, `editing_threshold`, `max_post_steps`, `eos_early_stop`), and decodes the full output sequence. Note: `dllm.pipelines.llada2.LLaDA2Sampler` is the intended cleaner API (see dllm section below) but is not currently used.

**Output format:** Both pipelines write JSONL with `{"file", "src", "tgt", "prediction"}` to `{autoregressive,diffusion}_models/results/<MODEL_NAME>_clinicalnlp_taskB_test1_full.jsonl`.

### `dllm` library (`/project/6101844/mattli/dllm`)

```
dllm/
├── core/
│   ├── samplers/   # BaseSampler → MDLMSampler, BD3LMSampler (inference algorithms)
│   ├── trainers/   # MDLMTrainer, BD3LMTrainer (extend HF Trainer)
│   ├── schedulers/ # Noise schedulers (Alpha/Kappa)
│   └── eval/       # Evaluation framework wrapping lm-evaluation-harness
├── pipelines/      # Model-specific code: llada2/, dream/, bert/, a2d/, editflow/, fastdllm/
├── utils/          # get_model, get_tokenizer, TerminalVisualizer, sample_trim, infill_trim
└── tools/          # CLI tools: merge adapters, download models/datasets, preprocess data
```

**LLaDA2.1-mini sampler:** Use `dllm.pipelines.llada2.LLaDA2Sampler` with `LLaDA2SamplerConfig`. Key config fields: `max_new_tokens`, `block_size`, `steps_per_block`, `temperature`, `top_p`, `threshold`.

**Infilling:** Pass masked inputs (using `tokenizer.mask_token`) with `add_generation_prompt=False`, call `sampler.infill()`, trim with `dllm.utils.infill_trim()`. Natural fit for SOAP note template filling where section headers are pre-filled.

**Model loading:** Always use `dllm.utils.get_model(model_args)` and `dllm.utils.get_tokenizer(model_args)` — these handle path resolution via `BASE_MODELS_DIR` env var.

## Code Guidelines

- Include a docstring at the top of each new Python file with how to run it.
- Reuse `dllm` utilities where possible; read existing scripts before writing new ones.
- Use absolute paths in documentation and scripts.
