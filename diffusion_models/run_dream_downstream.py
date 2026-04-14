"""
Downstream inference script for ACI-bench test1 using Dream-v0-Instruct-7B (diffusion model).

Loads Dream-v0-Instruct-7B via HuggingFace Transformers, runs clinical note generation on
the 40 examples in clinicalnlp_taskB_test1_full.json, and writes results to
results/Dream-v0-Instruct-7B_clinicalnlp_taskB_test1_full.jsonl.

Usage (called automatically by diffusion_inference.sbatch, or directly):
    python3 run_dream_downstream.py [--model_name_or_path PATH] [--max_new_tokens N] \
        [--steps N] [--temperature F] [--top_p F] [--data_dir DIR] [--output_dir DIR]
"""

import argparse
import json
import pathlib
import os

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"

DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. "
    "Given a doctor-patient dialogue, generate a structured clinical note."
)

CACHE_DIR = os.getenv("HF_HUB_CACHE", None)

if CACHE_DIR is None:
    print("Could not find cache directory")
    exit()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run downstream ACI-bench inference via Dream-v0-Instruct-7B (diffusion)"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH,
        help="HuggingFace model ID or local path to Dream-v0-Instruct-7B"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum number of new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--steps", type=int, default=512,
        help="Number of diffusion steps (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling top-p (default: 0.95)"
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

    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL_PATH,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    print("Model loaded.")

    data_path = pathlib.Path(args.data_dir) / "clinicalnlp_taskB_test1_full.json"
    examples = json.loads(data_path.read_text())["data"]
    print(f"Loaded {len(examples)} examples from {data_path}")

    output_path = (
        pathlib.Path(args.output_dir) / "Dream-v0-Instruct-7B_clinicalnlp_taskB_test1_full.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out_f:
        for i, ex in enumerate(examples, 1):
            try:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": ex["src"]},
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
                input_ids = inputs.input_ids.to(device="cuda")
                attention_mask = inputs.attention_mask.to(device="cuda")

                with torch.no_grad():
                    output = model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        output_history=False,
                        return_dict_in_generate=True,
                        steps=args.steps,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        alg="entropy",
                        alg_temp=0.,
                    )

                generations = [
                        tokenizer.decode(g[len(p):].tolist())
                    for p, g in zip(input_ids, output.sequences)
                ]
                prompt_tokens = attention_mask[0].sum().item()
                total_tokens = output.sequences[0].shape[0]
                generated_tokens = total_tokens - prompt_tokens

                print(f"prompt tokens: {prompt_tokens}")
                print(f"generated tokens: {generated_tokens}")
                print(f"total tokens: {total_tokens}")            


                prediction = generations[0].split(tokenizer.eos_token)[0]
                
                print(f"PREDICTION FROM DREAM: {prediction}")

            except Exception as e:
                import traceback
                print(f"  ERROR on example {i} ({ex['file']}): {e}")
                traceback.print_exc()
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
