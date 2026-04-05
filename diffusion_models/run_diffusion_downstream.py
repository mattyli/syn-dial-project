"""
Downstream inference script for ACI-bench test1 using LLaDA2.1-mini (diffusion model).

Loads LLaDA2.1-mini directly via the dllm library, runs clinical note generation on
the 40 examples in clinicalnlp_taskB_test1_full.json, and writes results to
results/LLaDA2.1-mini_clinicalnlp_taskB_test1_full.jsonl.

Usage (called automatically by diffusion_inference.sbatch):
    python3 run_diffusion_downstream.py [--model_name_or_path PATH] [--max_new_tokens N] \
        [--data_dir DIR] [--output_dir DIR]
"""

import argparse
import json
import pathlib

import torch

import dllm
import dllm.pipelines.llada2
from dllm.utils import ModelArguments, get_model, get_tokenizer, sample_trim

DEFAULT_MODEL_PATH = (
    "/home/mattli/projects/aip-zhu2048/mattli/hf_cache/hub/"
    "models--inclusionAI--LLaDA2.1-mini/snapshots/"
    "f21be037104f6e044e1a86b6d8864a6b85cc868e"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. "
    "Given a doctor-patient dialogue, generate a structured clinical note."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run downstream ACI-bench inference via LLaDA2.1-mini (diffusion)"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to LLaDA2.1-mini model snapshot"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4000,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--block_size", type=int, default=32,
        help="Diffusion block size (default: 32)"
    )
    parser.add_argument(
        "--steps_per_block", type=int, default=32,
        help="Denoising steps per block (default: 32)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature; 0.0 = greedy (default: 0.0)"
    )
    parser.add_argument(
        "--top_p", type=float, default=None,
        help="Nucleus sampling top-p (default: disabled)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Confidence threshold for token transfer (default: 0.95)"
    )
    parser.add_argument(
        "--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
        help="System message prepended to every user prompt"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/project/6101844/mattli/syn-dial-project/aci-bench-main/data/challenge_data_json",
        help="Directory containing the ACI-bench challenge data JSON files"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/project/6101844/mattli/syn-dial-project/diffusion_models/results",
        help="Directory to write inference results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_args = ModelArguments(model_name_or_path=args.model_name_or_path)

    print("Loading model and tokenizer...")
    model = get_model(model_args=model_args).eval()
    tokenizer = get_tokenizer(model_args=model_args)

    sampler = dllm.pipelines.llada2.LLaDA2Sampler(model=model, tokenizer=tokenizer)
    sampler_config = dllm.pipelines.llada2.LLaDA2SamplerConfig(
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        steps_per_block=args.steps_per_block,
        temperature=args.temperature,
        top_p=args.top_p,
        threshold=args.threshold,
    )

    data_path = pathlib.Path(args.data_dir) / "clinicalnlp_taskB_test1_full.json"
    examples = json.loads(data_path.read_text())["data"]
    print(f"Loaded {len(examples)} examples from {data_path}")

    output_path = pathlib.Path(args.output_dir) / "LLaDA2.1-mini_clinicalnlp_taskB_test1_full.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out_f:
        for i, ex in enumerate(examples, 1):
            try:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": ex["src"]},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                inputs = [input_ids]
                with torch.no_grad():
                    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
                sequences = sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
                prediction = sequences[0].strip()
            except Exception as e:
                print(f"  ERROR on example {i} ({ex['file']}): {e}")
                prediction = ""

            out_f.write(json.dumps({
                "file": ex["file"],
                "src": ex["src"],
                "tgt": ex["tgt"],
                "prediction": prediction,
            }) + "\n")
            out_f.flush()
            print(f"  [{i}/{len(examples)}] {ex['file']}")

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
