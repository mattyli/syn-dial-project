# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic dialogue generation project targeting EMNLP 2026. Uses two models:
- **Qwen2.5-0.5B-Instruct** (AR model) — weights at `~/../../model-weights/Qwen2.5-0.5B-Instruct`
- **LLaDA-8B-Instruct** (diffusion model) — via the `dllm` library at `/project/6101844/mattli/dllm`

## Environment Setup

Before running scripts, source shell config and activate the conda env:
```bash
source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
```

## Commands

**AR inference (Qwen):**
```bash
# Batch mode from JSONL file
python3 run_inference.py --input data/prompts.jsonl --output data/responses.jsonl

# Submit to Slurm
sbatch --export=ALL,INPUT=data/prompts.jsonl,OUTPUT=data/responses.jsonl run_inference.sh
```

**Diffusion inference (LLaDA via `sample.py`):**
```bash
python3 -u sample.py --model_name_or_path "/home/mattli/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
# Sampler options: --steps 128 --max_new_tokens 128 --block_size 32 --temperature 0.0 --remasking low_confidence
```

**GPU tasks via Slurm interactive:**
```bash
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python ...
```

**Tests (in dllm):**
```bash
cd /project/6101844/mattli/dllm && pytest scripts/tests -v -ra
```

## Input/Output Format

JSONL input — each line must have a `"prompt"` key:
```json
{"prompt": "Generate a short customer service dialogue about a lost package."}
```
Output lines have `"prompt"` and `"response"` keys. Logs go to `logs/`.

## Architecture

### This repo (`syn-dial-project`)
- `run_inference.py` — batch AR inference with Qwen; supports single prompt (`--prompt`) or JSONL file mode
- `run_inference.sh` — Slurm wrapper for `run_inference.py`
- `sample.py` — diffusion inference using `dllm.core.samplers.MDLMSampler`; demonstrates both batch sampling and fill-in-the-blanks (infilling)

### `dllm` library (`/project/6101844/mattli/dllm`)

The `dllm` package is the core diffusion LM library. Its key layers:

```
dllm/
├── core/
│   ├── samplers/   # BaseSampler → MDLMSampler, BD3LMSampler (inference algorithms)
│   ├── trainers/   # MDLMTrainer, BD3LMTrainer (extend HF Trainer)
│   ├── schedulers/ # Noise schedulers (Alpha/Kappa)
│   └── eval/       # Evaluation framework wrapping lm-evaluation-harness
├── pipelines/      # Model-specific code: llada/, dream/, bert/, a2d/, editflow/, fastdllm/
├── data/           # Dataset loaders (Alpaca, Ultrachat, OPC, S1K)
├── utils/          # get_model, get_tokenizer, TerminalVisualizer, sample_trim, infill_trim
└── tools/          # CLI tools: merge adapters, download models/datasets, preprocess data
```

**Sampler config flow:** `SamplerConfig` (dataclass) is parsed from CLI via `HfArgumentParser` and passed directly to `sampler.sample()` or `sampler.infill()`. All `MDLMSamplerConfig` fields (`steps`, `block_size`, `temperature`, `remasking`, `cfg_scale`, `stochastic_transfer`, etc.) are CLI args.

**Model loading:** Always use `dllm.utils.get_model(model_args)` and `dllm.utils.get_tokenizer(model_args)` — these handle path resolution via `BASE_MODELS_DIR` env var.

## Code Guidelines

- Include a docstring at the top of each new Python file with how to run it.
- Preview existing code before writing new code; reuse `dllm` utilities where possible.
- Use absolute paths in documentation and scripts.
