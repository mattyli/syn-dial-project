"""
Analyze tokenized sequence length distribution for MedSynth training examples.

Tokenizes the full chat-template-formatted prompt+completion for every training
example and reports max, mean, and percentile statistics.

Usage:
    python analyze_seq_lengths.py \
        [--data_path /project/aip-zhu2048/mattli/syn-dial-project/MedSynth_huggingface_final.csv] \
        [--num_samples 1000] \
        [--model_name meta-llama/Meta-Llama-3.1-8B-Instruct]
"""

import argparse
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import load_medsynth
from utils.constants import DEFAULT_BASE_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence length distribution analysis")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/project/aip-zhu2048/mattli/syn-dial-project/MedSynth_huggingface_final.csv",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model name or local path for the tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading dataset: {args.data_path} (num_samples={args.num_samples})")
    dataset = load_medsynth(args.data_path, num_samples=args.num_samples, seed=args.seed)
    print(f"  {len(dataset)} examples")

    print("Tokenizing ...")
    lengths = []
    for example in dataset:
        ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))

    lengths = np.array(lengths)
    pct = [50, 75, 90, 95, 99]
    percentiles = np.percentile(lengths, pct).astype(int)

    print(f"\n{'='*50}")
    print(f"  N examples : {len(lengths):,}")
    print(f"  Max length : {lengths.max():,}")
    print(f"  Mean length: {lengths.mean():,.1f}")
    print(f"  Std dev    : {lengths.std():,.1f}")
    for p, v in zip(pct, percentiles):
        print(f"  p{p:<2}        : {v:,}")
    print(f"  Min length : {lengths.min():,}")
    print(f"{'='*50}")

    over_8192 = (lengths > 8192).sum()
    print(f"\n  Examples > 8192 tokens (max_seq_length): {over_8192} ({100*over_8192/len(lengths):.1f}%)")


if __name__ == "__main__":
    main()
