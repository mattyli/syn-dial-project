import os
import uuid
from argparse import ArgumentParser
from collections import defaultdict, Counter
from pathlib import Path

import ipdb
import jsonlines
from tqdm import tqdm

from generation_modules.vllm_model import VLLMGenerator
from generation_modules.generate_model_util import save_to_file


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)

    args = parser.parse_args()

    if args.model_name == "Qwen2.5-72B":
        args.full_model_name = "Qwen/Qwen2.5-72B-Instruct"
        args.batch_size = 256
        args.max_model_len, args.max_gen_len = 4096, 2048

        # number of solution to generate per problem
        args.num_solutions_per_problem = 3

        # minimum ratio of majority answers to be considered to be `correct`
        args.min_majority_ratio = 0.5

    elif args.model_name == "R1-32B":
        args.full_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        args.batch_size = 64
        args.max_model_len, args.max_gen_len = 32768, 20000

        # number of solution to generate per problem
        args.num_solutions_per_problem = 3

        # minimum ratio of majority answers to be considered to be `correct`
        args.min_majority_ratio = 0.5

    else:
        raise NotImplementedError

    args.out_filename = Path(args.output_filename)
    os.makedirs(args.out_filename.parent, exist_ok=True)
    assert not args.out_filename.exists(), f"{args.out_filename} already exists."

    return args


if __name__ == "__main__":
    args = parse_args()

    with jsonlines.open(args.input_filename) as f:
        samples = list(f)

    model = VLLMGenerator(args.full_model_name, args.max_model_len, args.max_gen_len)

    for batch_start_idx in tqdm(range(0, len(samples), args.batch_size)):
        batch_samples = samples[batch_start_idx:batch_start_idx + args.batch_size]

        # -- run VLLM to generate solutions -- #
        result_list = model.batch_prompt_solution(batch_samples, args.num_solutions_per_problem)

        # -- aggregate results per each problem id -- #
        batch_result_list = defaultdict(list)  # {"problem_id1": [result1, result2, ...], ...}
        for result in result_list:
            batch_result_list[result['prompt_id']].append(result)

        # -- filter out based on majority answer -- #
        out_batch_samples = []
        for problem_id, result_list in batch_result_list.items():
            # find the majority answer's # of occurrences
            answers = [result["answer"] for result in result_list]
            answer_counter = Counter(answers)

            majority_size = max(answer_counter.values())

            if majority_size >= int(len(answers) * args.min_majority_ratio):
                for answer in answer_counter.keys():
                    # find the majority answer that is not None
                    if answer is not None and answer_counter[answer] == majority_size:
                        # save all solutions that have the majority answer
                        for result in result_list:
                            if result["answer"] == answer:
                                out_batch_samples.append({
                                    "prompt": result["prompt"],
                                    "prompt_id": result["prompt_id"],
                                    "completion": result["completion"],
                                    "id": f"{'.'.join(result['prompt_id'].split('.')[:-1])}.{uuid.uuid4().hex}",
                                })

        save_to_file(out_batch_samples, args.out_filename, save_mode="a")



