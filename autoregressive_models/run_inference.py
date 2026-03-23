"""
Inference script for Qwen2.5-0.5B-Instruct.
Reads prompts from a JSONL file (one {"prompt": "..."} per line) and writes
completions to an output JSONL file.
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#MODEL_PATH = "/model-weights/Qwen2.5-3B"
MODEL_PATH = "home/mattli/aip-zhu2048/projects/local-models/LLaDA-8B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="A single prompt string to run inference on (prints response to stdout)"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to input JSONL file. Each line: {\"prompt\": \"...\"}"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output JSONL file."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum number of new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p nucleus sampling (default: 0.9)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size (default: 8)"
    )
    parser.add_argument(
        "--system_prompt", type=str,
        default="You are a helpful assistant.",
        help="System prompt to prepend to every user message"
    )
    args = parser.parse_args()
    if args.prompt is None and args.input is None:
        parser.error("either --prompt or --input/--output must be provided")
    if args.input is not None and args.output is None:
        parser.error("--output is required when --input is provided")
    return args


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON on line {lineno}: {e}", file=sys.stderr)
                continue
            if "prompt" not in obj:
                print(f"Warning: skipping line {lineno}, missing 'prompt' key", file=sys.stderr)
                continue
            prompts.append(obj)
    return prompts


def build_chat_text(tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_PATH} on {device}...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.", file=sys.stderr)

    records = load_prompts(args.input)
    print(f"Loaded {len(records)} prompts from {args.input}", file=sys.stderr)

    with open(args.output, "w") as out_f:
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start : batch_start + args.batch_size]
            texts = [
                build_chat_text(tokenizer, args.system_prompt, r["prompt"])
                for r in batch
            ]

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            for record, output_ids in zip(batch, outputs):
                generated = tokenizer.decode(
                    output_ids[input_len:], skip_special_tokens=True
                )
                result = {**record, "response": generated}
                out_f.write(json.dumps(result) + "\n")

            completed = min(batch_start + args.batch_size, len(records))
            print(f"  {completed}/{len(records)} done", file=sys.stderr)

    print(f"Results written to {args.output}", file=sys.stderr)


def run_single_prompt(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_PATH} on {device}...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.", file=sys.stderr)

    text = build_chat_text(tokenizer, args.system_prompt, args.prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(generated)


if __name__ == "__main__":
    args = parse_args()
    if args.prompt is not None:
        run_single_prompt(args)
    else:
        run_inference(args)
