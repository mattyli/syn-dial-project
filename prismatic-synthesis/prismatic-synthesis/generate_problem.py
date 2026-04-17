import json
import os
import random
import uuid
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List

import ipdb
import jsonlines
import numpy as np
from tqdm import tqdm

from generation_modules.vllm_model import VLLMGenerator
from generation_modules.generate_model_util import save_to_file


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)

    parser.add_argument("--target_size", type=int, default=3000)
    parser.add_argument("--num_fewshot_samples", type=int, default=5)
    parser.add_argument("--num_new_problems", type=int, default=2)  # actual # of generations - num_new_problems * num_new_problems
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    args.input_filename = Path(args.input_filename)

    if args.model_name == "Qwen2.5-72B":
        args.full_model_name = "Qwen/Qwen2.5-72B-Instruct"
    else:
        raise NotImplementedError

    # prepare out_filename
    args.out_filename = Path(args.output_filename)
    os.makedirs(args.out_filename.parent, exist_ok=True)
    assert not args.out_filename.exists(), f"{args.out_filename} already exists."

    return args


if __name__ == "__main__":
    args = parse_args()

    with jsonlines.open(args.input_filename) as f:
        samples = list(f)
        sample_levels = np.array([s["level"] if "level" in s else 1 for s in samples])
        sample_levels = sample_levels / np.sum(sample_levels)

    model = VLLMGenerator(args.full_model_name, max_model_len=8192, max_gen_len=8192)

    num_generated = 0
    last_fewshot_sample_idx = 0  # if we only use 1 fewshot sample, retain the last index of used fewshot sample
    with tqdm(total=args.target_size, initial=num_generated) as pbar:
        while num_generated < args.target_size:
            # -- prepare batched_fewshot_samples, weighted by the sample level -- #
            batched_fewshot_indices = [
                np.random.choice(range(len(samples)), size=args.num_fewshot_samples, replace=False, p=sample_levels)
                for _ in range(args.batch_size)
            ]
            batched_fewshot_samples = [
                [samples[idx] for idx in fewshot_indices] for fewshot_indices in batched_fewshot_indices
            ]

            # -- generate new problems (ideally generate `num_new_problems` * `batch_size` problems) -- #
            batch_out_samples = model.batch_prompt_problem(batched_fewshot_samples, args.num_new_problems)

            # -- add `id` and format solution to include Final Answer -- #
            for fewshot_samples, out_samples in zip(batched_fewshot_samples, batch_out_samples):
                for out_sample in out_samples:
                    out_sample["prompt_id"] = f"gen.{args.input_filename.stem}.{uuid.uuid4().hex}"

                # -- save to file -- #
                save_to_file(out_samples, args.out_filename, save_mode="a")
                num_generated += len(out_samples)

            pbar.n = num_generated
            pbar.refresh()

