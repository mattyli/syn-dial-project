# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic dialogue generation project (targeting EMNLP 2026) focused on **diversity and data selection** within the MedSynth dataset. Diffusion models are no longer in scope — the project is exclusively AR-based.

The core pipeline:
1. **Diversity selection** (`prismatic-synthesis/`): G-Vendi gradient-based clustering to select diverse subsets of `MedSynth_huggingface_final.csv`.
2. **Dialogue → Note finetuning** (`eval/`): QLoRA-finetune an AR model on selected MedSynth Dialogue→Note pairs.
3. **Benchmarking** (`eval/benchmarking/`, `aci-bench-main/`): Evaluate finetuned models on ACI-bench test1 (`clinicalnlp_taskB_test1_full`, 40 examples).
4. **Direct AR inference** (`autoregressive_models/`): Zero-shot/few-shot inference via vec-inf (vLLM) without finetuning.

Models: Meta-Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and others — served via `vec-inf` (vLLM) on Slurm.

## Environment Setup

Activate the project venv before running any scripts:
```bash
source ~/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"
```

The project directory is symlinked: `/project/6101844/mattli/syn-dial-project` ↔ `/project/aip-zhu2048/mattli/syn-dial-project`. Slurm scripts use the `aip-zhu2048` path.

## Commands

**QLoRA finetuning (Slurm):**
```bash
cd eval
# Finetune on a diversity-selected subset
bash run_finetune.sh \
  --job_name finetune-diverse-k100-n1000 \
  --data_path /project/aip-zhu2048/mattli/syn-dial-project/prismatic-synthesis/results/selected_subset_N1000_K100.jsonl

# Finetune on full MedSynth (no --data_path defaults to MedSynth_huggingface_final.csv)
bash run_finetune.sh --job_name finetune-full-medsynth

# Other options: --num_samples, --base_model, --epochs, --lora_r, --seed
# Outputs: eval/results/<job_name>/slurm.out|err + model/final_model/
```

**Benchmarking a finetuned model:**
```bash
cd eval/benchmarking
bash run_benchmarking.sh --run_name <job_name>
# Loads adapter from eval/results/<job_name>/model/final_model/
# Runs inference on ACI-bench test1 (40 examples), computes BLEU/ROUGE/BERTScore/METEOR
# Results: eval/results/<job_name>/benchmark_results.json
```

**Direct AR inference (via vec-inf + Slurm):**
```bash
cd autoregressive_models
bash run_autoregressive_inference.sh Meta-Llama-3.1-8B-Instruct
# No arg = launches all four target models in parallel:
#   medgemma-27b-text-it, gemma-4-26B-A4B-it, Qwen3.5-27B, aya-expanse-32b
# Allowed models: Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
#   Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B, gpt-oss-20b,
#   medgemma-27b-text-it, gemma-4-26B-A4B-it, Qwen3.5-27B, aya-expanse-32b
# Output: autoregressive_models/results/<MODEL_NAME>_clinicalnlp_taskB_test1_full.jsonl
```

**ACI-bench metric evaluation:**
```bash
cd aci-bench-main/evaluation
python3 evaluate_fullnote.py <gold_csv> <pred_file> [metadata_csv]
# Results written to aci-bench-main/evaluation/results/<pred_filename>.json
# Note: hardcodes CUDA_VISIBLE_DEVICES=1; run on a GPU node
```

**Diversity subset selection:**
```bash
cd prismatic-synthesis
bash run_pipeline.sh  # See script for gradient collection + select_diverse_subset.py args
# select_diverse_subset.py: K-means on gradient vectors → inverse-frequency weighted sampling
# Output: results/selected_subset_N{N}_K{k}.jsonl + _metadata.json + cluster_plot.png
```

## Architecture

### Data flow

```
MedSynth_huggingface_final.csv
    ↓ [prismatic-synthesis/select_diverse_subset.py]  G-Vendi K-means on gradient vectors
    ↓
selected_subset_N{N}_K{k}.jsonl
    ↓ [eval/finetune.py]  QLoRA 4-bit, SFTTrainer
    ↓
eval/results/<run-name>/model/final_model/  (LoRA adapter)
    ↓ [eval/benchmarking/run_benchmarking.py]
    ↓
benchmark_results.json  (BLEU, ROUGE, BERTScore, METEOR)
```

Alternative (zero-shot, no finetuning):
```
ACI-bench test1  →  [autoregressive_models/run_downstream.py + vec-inf]  →  results/*.jsonl
```

### Key components

**`autoregressive_models/`** — Direct inference without finetuning. `run_autoregressive_inference.sh` calls `vec-inf launch <MODEL>` to get a Slurm job ID for the vLLM server, then submits `downstream_job.sbatch` with `--dependency=after:<job_id>`. `run_downstream.py` waits via `VecInfClient.wait_until_ready()`, hits the OpenAI-compatible endpoint (temperature=0.1, max_tokens=512), writes JSONL `{file, src, tgt, prediction}`, then scancels the server job.

**`eval/`** — Finetuning + benchmarking pipeline.
- `finetune.py`: QLoRA via `unsloth` (`FastLanguageModel`), LoRA r=16 on all attention+MLP projections, bf16, batch=2, grad_accum=4, lr=2e-4. Default base model: `unsloth/Meta-Llama-3.1-8B-Instruct`.
- `utils/dataset.py`: Loads CSV (`Note`/`Dialogue` columns) or JSONL (`prompt`/`completion`), formats as Llama-3 chat turns.
- `utils/constants.py`: LoRA config, system prompt (SOAP format), training hyperparameters.
- `benchmarking/run_benchmarking.py`: Loads adapter, runs inference on ACI-bench test1, optionally runs LLM judge (Prometheus or GPT-4) for aspect-level scoring or pairwise comparison.

**`prismatic-synthesis/`** — Diversity selection.
- `select_diverse_subset.py`: Loads precomputed 1024-D gradient vectors, runs K-means (cosine similarity, 20 Lloyd iterations), weights samples by inverse cluster size, samples N examples without replacement.
- `g-vendi/`: G-Vendi scoring (gradient collection + entropy computation).

**`aci-bench-main/`** — Benchmark dataset (train=67, valid=20, test1=40, test2=40, test3=40 examples) and evaluation scripts. Primary target is `test1` (`clinicalnlp_taskB_test1_full`).

## Code Guidelines

- Include a docstring at the top of each new Python file with how to run it.
- Read existing scripts before writing new ones.
- Use absolute paths in documentation and scripts.
