import random
from typing import Dict, List

import ipdb
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import math_verify

from generation_modules.generate_model_util import Answer
from generation_modules.prompts import *


class VLLMGenerator:
    def __init__(self, model_name: str, max_model_len: int, max_gen_len: int):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.max_gen_len = max_gen_len

        self.llm = LLM(model=self.model_name, tensor_parallel_size=8, max_model_len=self.max_model_len, swap_space=16, enable_prefix_caching=True, seed=random.randint(0, 99999))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def batch_prompt_problem(self, batch_fewshot_samples: List[List[Dict]], num_new_problems: int) -> List[List[Dict]]:
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=0.95,
            max_tokens=self.max_gen_len,
            seed=random.randint(0, 99999),
        )

        # -- prepare prompt and run VLLM -- #
        prompt_list = [
            self.prepare_problem_prompt(fewshot_samples, num_new_problems)
            for fewshot_samples in batch_fewshot_samples
        ]
        output_list = self.llm.generate(prompt_list, sampling_params, use_tqdm=False)

        # -- parse outputs into problems -- #
        batch_problems = []
        for sample_output in output_list:
            problems = []
            for raw_generation in sample_output.outputs:
                problems += self.parse_problem_from_generation(raw_generation.text)
            batch_problems.append(problems)

        return batch_problems

    def prepare_problem_prompt(self, fewshot_samples: List[Dict], num_new_problems: int) -> str:
        instruction_prompt = problem_instruction_template.replace("$#$num_new_problems$#$", str(num_new_problems))

        # -- prepare fewshot sample prompts -- #
        fewshot_sample_prompts = [self.prepare_problem_fewshot_sample_prompt(sample) for sample in fewshot_samples]
        random.shuffle(fewshot_sample_prompts)  # shuffle few-shot examples for better diversity
        fewshot_sample_prompts = "\n\n".join(fewshot_sample_prompts)

        # -- prepare full_prompt by applying chat template -- #
        messages = [
            {"role": "user", "content": instruction_prompt + fewshot_sample_prompts}
        ]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return full_prompt

    @staticmethod
    def prepare_problem_fewshot_sample_prompt(sample: Dict) -> str:
        # -- prepare prompt -- #
        fewshot_sample_prompt = problem_fewshot_template.replace("$#$problem$#$", sample["prompt"])

        return fewshot_sample_prompt

    @staticmethod
    def parse_problem_from_generation(generation: str) -> List[Dict]:
        out_samples = []
        for split_generation in generation.split("---")[1:-1]:
            split_generation = split_generation.strip()

            if all(tag in split_generation for tag in ["[[Problem]]"]):
                problem = split_generation[split_generation.find("[[Problem]]") + len("[[Problem]]"):]

                out_samples.append({
                    "prompt": problem.strip(),
                })

        return out_samples

    def batch_prompt_solution(self, samples: List[Dict], num_solutions_per_problem: int) -> List[Dict]:
        """
        Generate `num_solutions_per_problem` solutions per each sample in `samples`.
        Return: A list of dict formatted as
        {
        "prompt": problem in sample
        "prompt_id": problem_id in sample
        "completion": solution generated from model
        "answer": answer generated from model
        }
        """
        sampling_params = SamplingParams(
            n=num_solutions_per_problem,
            temperature=0.75,
            top_p=0.95,
            max_tokens=self.max_gen_len,
            seed=random.randint(0, 65535),
        )

        # -- prepare prompt and run VLLM -- #
        prompt_list = [
            self.prepare_solution_prompt(sample["prompt"]) for sample in samples
        ]
        output_list = self.llm.generate(prompt_list, sampling_params, use_tqdm=False)

        # -- parse outputs into out_samples -- #
        out_samples = []
        for sample, sample_output in zip(samples, output_list):
            for raw_generation in sample_output.outputs:
                out_samples.append(self.parse_solution_from_generation(sample, raw_generation.text))

        return out_samples

    def prepare_solution_prompt(self, problem: str) -> str:
        prompt = solution_instruction_template.replace("$#$problem$#$", problem)

        # -- prepare full_prompt by applying chat template -- #
        messages = [
            {"role": "system", "content": solution_instruction_system_template},
            {"role": "user", "content": prompt}
        ]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return full_prompt

    @staticmethod
    def parse_solution_from_generation(sample: Dict, generation: str) -> Dict:
        try:
            parsed_answer = Answer(math_verify.parse(generation))

            # if the answer is too long, just make it None (or it leads to parsing error)
            if len(str(parsed_answer)) > 300:
                parsed_answer = None
            # if boxed not included in the generation, just make answer None
            if "boxed" not in generation:
                parsed_answer = None

        except Exception:
            parsed_answer = None

        return {
            "sample": sample,
            "prompt": sample["prompt"],
            "prompt_id": sample["prompt_id"] if "prompt_id" in sample else None,
            "completion": generation,
            "answer": parsed_answer,
        }


