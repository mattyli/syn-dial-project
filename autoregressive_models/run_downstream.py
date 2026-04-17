"""
Downstream inference script for ACI-bench test1 (clinicalnlp_taskB_test1_full).

Waits for a vec-inf server to be ready, then runs clinical note generation on
the test1 set and writes results to results/<MODEL_NAME>_clinicalnlp_taskB_test1_full.jsonl.

Usage (called automatically by downstream_job.sbatch):
    python3 run_downstream.py <SERVER_JOB_ID> <MODEL_NAME> [--max_tokens N] [--temperature T] [--top_p P] [--system_prompt "..."]
"""

import argparse
import json
import pathlib

from openai import OpenAI
from vec_inf.client import VecInfClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run downstream ACI-bench inference via a vec-inf server"
    )
    parser.add_argument("--job_id", type=str, help="SLURM job ID of the vec-inf server")
    parser.add_argument("--model_name", type=str, help="Model name served by the vec-inf server")
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Top-p nucleus sampling (default: 1.0)"
    )
    parser.add_argument(
        "--system_prompt", type=str,
        default=(
            "Assume you are a very experienced physician and you are conducting research. "
            "The research project is to generate synthetic medical notes from doctor-patient conversations. "
            "The notes must be in this format:\n"
            "** 1. Subjective: This section includes the patient's own description of their symptoms and complaints. "
            "Roll a dice, if the result is odd, break this part down into several sub-parts like "
            "Chief Complaint (CC), History of Present Illness (HPI), Review of Systems (ROS).\n"
            "** 2. Objective: This section includes observations and data gathered by the physician, such as vital "
            "signs, physical examination findings, and test results.\n"
            "** 3. Assessment: This section includes the physician's evaluation of the patient's condition, including "
            "a diagnosis or differential diagnosis.\n"
            "** 4. Plan: This section includes the physician's recommendations for treatment, management, and follow-up.\n\n"
            "You will be given a scenario containing your role. Your role can be a Family Medicine Physician, a "
            "General physician, or a specialist with different specialties. Your task is to generate the note based on "
            "the scenario. The note you generate must be in the format mentioned above. "
            "Output only the medical note with no preamble, introduction, or closing remarks."
        ),
        help="System prompt to prepend to every user message"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/project/6101844/mattli/syn-dial-project/aci-bench-main/data/challenge_data_json",
        help="Directory containing the ACI-bench challenge data JSON files"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/project/6101844/mattli/syn-dial-project/autoregressive_models/results",
        help="Directory to write inference results (must be on shared filesystem)"
    )
    return parser.parse_args()


args = parse_args()

# --- Wait for the server to be ready (mirrors the VectorInstitute example)
vi_client = VecInfClient()
print(f"Waiting for SLURM job {args.job_id} to be ready...")
status = vi_client.wait_until_ready(slurm_job_id=args.job_id)
print(f"Server ready at {status.base_url}")

# --- OpenAI-compatible client pointing at the vec-inf server
api_client = OpenAI(base_url=status.base_url, api_key="EMPTY")

# --- Load test1 data
DATA_PATH = pathlib.Path(args.data_dir) / "clinicalnlp_taskB_test1_full.json"
examples = json.loads(DATA_PATH.read_text())["data"]
print(f"Loaded {len(examples)} examples from {DATA_PATH}")

OUTPUT_PATH = pathlib.Path(args.output_dir) / args.model_name / f"{args.model_name}_clinicalnlp_taskB_test1_full.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w") as out_f:
    for i, ex in enumerate(examples, 1):
        response = api_client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": ex["src"]},
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        prediction = response.choices[0].message.content
        out_f.write(json.dumps({
            "file": ex["file"],
            "src": ex["src"],
            "tgt": ex["tgt"],
            "prediction": prediction,
        }) + "\n")
        print(f"  [{i}/{len(examples)}] {ex['file']}")

print(f"Results written to {OUTPUT_PATH}")
