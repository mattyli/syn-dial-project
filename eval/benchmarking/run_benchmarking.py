"""
Benchmarking pipeline for finetuned Dialogue→Note models on ACI-bench test set.

Loads a LoRA-finetuned model from eval/results/<run-name>/model/final_model/,
runs inference on the 40-example ACI-bench test set, then computes:
  - Traditional metrics: BLEU, ROUGE-1/2/L/LSum, BERTScore, METEOR
  - LLM jury: absolute per-aspect scoring (1–5) and optional pairwise relative
    preference between two runs.

Supported judges (--judge):
  prometheus   Prometheus 7B v2.0, in-process vLLM (default)
  gemma4_26b   gemma-4-26B-A4B-it, in-process vLLM (requires 2 GPUs)
  qwen35_27b   Qwen3.5-27B, in-process vLLM (requires 2 GPUs)

After running all three judges, use compute_majority_vote.py to aggregate.

Uses vendor/MedSynth/eval/utils/ directly (added to sys.path).

Usage:
    python run_benchmarking.py --run-name medsynth-full [options]
    python run_benchmarking.py --run-name medsynth-full --run-name-b medsynth-random-n5000
    python run_benchmarking.py --run-name medsynth-full --skip-llm-judge  # CPU-safe
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import re
import sys

import torch

# ── vendor path ───────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "MedSynth" / "eval"))

# vendor imports deferred to function bodies so unsloth can patch transformers first

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_SLURM = pathlib.Path("/project/aip-zhu2048/mattli/syn-dial-project")

HF_HUB_CACHE = pathlib.Path("/home/mattli/projects/aip-zhu2048/mattli/hf_cache")
PROMETHEUS_MODEL_PATH = str(
    HF_HUB_CACHE
    / "models--prometheus-eval--prometheus-7b-v2.0"
    / "snapshots"
    / "66ffb1fc20beebfb60a3964a957d9011723116c5"
)
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "aci-bench-main"
    / "data"
    / "challenge_data_json"
    / "clinicalnlp_taskB_test1_full.json"
)
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

LARGE_JUDGE_MODEL_PATHS = {
    "gemma4_26b": "/model-weights/gemma-4-26B-A4B-it",
    "qwen35_27b": "/model-weights/Qwen3.5-27B",
}

DIAL2NOTE_SYSTEM_PROMPT = (
    "You are an assistant for medical professionals, specializing in summarizing their "
    "conversations with patients. Your role is to accurately and comprehensively summarize "
    "these conversations in the SOAP (Subjective, Objective, Assessment, Plan) format. "
    "Ensure that each summary is thorough and precise, reflecting all relevant details from "
    "the conversation to provide a reliable medical record."
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark finetuned model on ACI-bench test set")
    p.add_argument("--run-name", required=True,
                   help="Name matching eval/results/<run-name>/model/final_model/")
    p.add_argument("--model-path", default=None,
                   help="Explicit path to LoRA adapter dir (overrides --run-name lookup)")
    p.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct",
                   help="Base model for FastLanguageModel (must match training base)")
    p.add_argument("--run-name-b", default=None,
                   help="Second run name for pairwise relative jury")
    p.add_argument("--model-path-b", default=None,
                   help="Explicit path for second model (overrides --run-name-b lookup)")
    p.add_argument(
        "--judge",
        choices=["prometheus", "gemma4_26b", "qwen35_27b"],
        default="prometheus",
        help=(
            "LLM judge backend. "
            "'prometheus' = Prometheus 7B v2.0, in-process vLLM (default). "
            "'gemma4_26b' = gemma-4-26B-A4B-it, in-process vLLM (requires 2 GPUs). "
            "'qwen35_27b' = Qwen3.5-27B, in-process vLLM (requires 2 GPUs). "
            "Run all three, then use compute_majority_vote.py for aggregation."
        ),
    )
    p.add_argument("--skip-auto-metrics", action="store_true")
    p.add_argument("--skip-llm-judge", action="store_true")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="Root directory for benchmark outputs")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                   help="Path to clinicalnlp_taskB_test1_full.json")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


# ── data helpers ──────────────────────────────────────────────────────────────

def load_aci_test(data_path: str) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)["data"]


def _predictions_cache_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / "predictions.jsonl"


def save_predictions(output_dir: pathlib.Path, examples: list[dict], predictions: list[str]) -> None:
    cache = _predictions_cache_path(output_dir)
    with open(cache, "w") as f:
        for ex, pred in zip(examples, predictions):
            f.write(json.dumps({"file": ex["file"], "src": ex["src"], "tgt": ex["tgt"], "prediction": pred}) + "\n")
    print(f"Predictions saved → {cache}")


def load_predictions_cache(output_dir: pathlib.Path) -> list[dict] | None:
    cache = _predictions_cache_path(output_dir)
    if not cache.exists():
        return None
    rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} cached predictions from {cache}")
    return rows


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model_path: str,
    examples: list[dict],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Load finetuned LoRA model and generate notes for all examples."""
    import unsloth  # noqa: F401 — must import before transformers
    from unsloth import FastLanguageModel

    print(f"Loading model from {model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Gemma3 loads a Processor (has image_processor); Llama/Qwen load a plain Tokenizer.
    # Processors require content as [{"type": "text", "text": "..."}]; plain tokenizers want a string.
    use_typed_content = hasattr(tokenizer, "image_processor")

    predictions: list[str] = []
    for i, ex in enumerate(examples):
        user_text = f"This is the conversation: {ex['src']}"
        sys_content  = [{"type": "text", "text": DIAL2NOTE_SYSTEM_PROMPT}] if use_typed_content else DIAL2NOTE_SYSTEM_PROMPT
        user_content = [{"type": "text", "text": user_text}]               if use_typed_content else user_text
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user",   "content": user_content},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        generated = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        predictions.append(generated.strip())
        print(f"  [{i + 1}/{len(examples)}] {ex['file']}")

    # free GPU memory before prometheus loads its own vLLM instance
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


# ── auto metrics ──────────────────────────────────────────────────────────────

def compute_auto_metrics(
    predictions: list[str],
    references: list[str],
    output_dir: pathlib.Path,
    run_name: str,
) -> dict:
    from utils.automatic_metrics import MetricsComputer

    print("Computing automatic metrics ...")
    # BLEU expects references as list-of-lists (one list of refs per example)
    references_bleu = [[r] for r in references]
    mc = MetricsComputer(prediction_list=predictions, gt_list=references)
    mc_bleu = MetricsComputer(prediction_list=predictions, gt_list=references_bleu)
    rouge = mc.compute_ROUGE()
    results = {
        "run_name": run_name,
        "num_examples": len(predictions),
        "BLEU": mc_bleu.compute_BLEU(),
        "ROUGE-1": rouge["rouge1"],
        "ROUGE-2": rouge["rouge2"],
        "ROUGE-L": rouge["rougeL"],
        "ROUGE-LSum": rouge["rougeLsum"],
        "BERTScore": mc.compute_BERTScore(),
        "METEOR": mc.compute_METEOR(),
    }
    out_path = output_dir / "auto_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Auto metrics → {out_path}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return results


# ── LLM jury ─────────────────────────────────────────────────────────────────

def _parse_result_score(text: str) -> int:
    """Extract integer 1–5 from '[RESULT] X' pattern. Returns -1 on failure."""
    m = re.search(r'\[RESULT\]\s*([1-5])', text)
    if m:
        return int(m.group(1))
    # Fallback: trailing digit at end of response
    m = re.search(r'\b([1-5])\b\s*$', text.strip())
    if m:
        return int(m.group(1))
    return -1


def _parse_relative_result(text: str) -> str:
    """Extract 'A' or 'B' from '[RESULT] A/B'. Returns '' on failure."""
    m = re.search(r'\[RESULT\]\s*([AB])', text, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _load_large_judge(model_path: str) -> tuple:
    """Load a large judge model with NF4 bitsandbytes quantization. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading judge from {model_path} (NF4, double quant, BF16 compute) ...")
    # use_fast=False: Gemma fast tokenizer errors when extra_special_tokens is a list not a dict
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def _generate_judge_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def run_large_model_absolute_jury(
    conversations: list[str],
    references: list[str],
    predictions: list[str],
    run_name: str,
    output_dir: pathlib.Path,
    judge_type: str,
) -> None:
    """Score all aspects using a large model loaded with NF4 bitsandbytes quantization."""
    import utils.prometheus as vendor_prometheus
    from utils import constants as vendor_constants
    from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, ABS_SYSTEM_PROMPT

    model_path = LARGE_JUDGE_MODEL_PATHS[judge_type]
    base_name = f"Absolute_{judge_type}_scores"

    model, tokenizer = _load_large_judge(model_path)

    for aspect, rubric_data in vendor_constants.prometheus_absolute_rubric_data.items():
        print(f"  [{judge_type}] scoring aspect: {aspect}")
        rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

        scores: dict[int, dict] = {}
        for idx, (conv, ref, pred) in enumerate(zip(conversations, references, predictions)):
            instruction = vendor_constants.prometheus_absolute_instruction.format(
                conversation=conv, gt_note=ref,
            )
            prompt = ABSOLUTE_PROMPT.format(
                instruction=instruction,
                response=pred,
                reference_answer=ref,
                rubric=rubric,
            )
            messages = [
                {"role": "system", "content": ABS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            text = _generate_judge_response(model, tokenizer, formatted)
            score = _parse_result_score(text)
            scores[idx] = {
                "conversation": conv,
                "reference_note": ref,
                "model_note": pred,
                "feedback": text,
                "Score": score,
            }
            print(f"    [{idx + 1}/{len(conversations)}] score={score}")

        vendor_prometheus._save_prometheus_absolute_scores(
            prometheus_scores=scores,
            model_A_name=run_name,
            aspect=aspect,
            path=str(output_dir),
            base_name=base_name,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{judge_type}] absolute jury complete.")


def run_large_model_relative_jury(
    conversations: list[str],
    references: list[str],
    predictions_a: list[str],
    predictions_b: list[str],
    run_name_a: str,
    run_name_b: str,
    output_dir: pathlib.Path,
    judge_type: str,
) -> None:
    """Pairwise relative scoring using a large model loaded with NF4 bitsandbytes quantization."""
    import utils.prometheus as vendor_prometheus
    from utils import constants as vendor_constants
    from prometheus_eval.prompts import RELATIVE_PROMPT, REL_SYSTEM_PROMPT

    model_path = LARGE_JUDGE_MODEL_PATHS[judge_type]

    model, tokenizer = _load_large_judge(model_path)

    preferences: dict[int, dict] = {}
    for idx, (conv, ref, pred_a, pred_b) in enumerate(
        zip(conversations, references, predictions_a, predictions_b)
    ):
        instruction = vendor_constants.prometheus_preference_instruction.format(
            conversation=conv, ground_truth_note=ref,
        )
        prompt = RELATIVE_PROMPT.format(
            instruction=instruction,
            response_A=pred_a,
            response_B=pred_b,
            reference_answer=ref,
            rubric=vendor_constants.prometheus_preference_rubric,
        )
        messages = [
            {"role": "system", "content": REL_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = _generate_judge_response(model, tokenizer, formatted)
        pref = _parse_relative_result(text)
        preferences[idx] = {
            "conversation": conv,
            "reference_note": ref,
            "model_A_note": pred_a,
            "model_B_note": pred_b,
            "feedback": text,
            "Preference": pref,
        }
        print(f"  [{idx + 1}/{len(conversations)}] preference={pref}")

    vendor_prometheus._save_prometheus_relative_scores(
        prometheus_scores=preferences,
        model_A_name=run_name_a,
        base_name=f"relative_{judge_type}",
        model_B_name=run_name_b,
        path=str(output_dir),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{judge_type}] relative jury complete.")


def _load_prometheus_judge(mode: str):
    """Load a PrometheusEval judge using the local Prometheus 7B snapshot."""
    from prometheus_eval import PrometheusEval
    from prometheus_eval.vllm import VLLM
    from prometheus_eval.prompts import ABSOLUTE_PROMPT, RELATIVE_PROMPT

    n_gpus = torch.cuda.device_count()
    model = VLLM(model=PROMETHEUS_MODEL_PATH, gpu_memory_utilization=0.85, tensor_parallel_size=n_gpus)
    kwargs = {}
    if mode == "absolute":
        kwargs["absolute_grade_template"] = ABSOLUTE_PROMPT
    else:
        kwargs["relative_grade_template"] = RELATIVE_PROMPT
    return PrometheusEval(model=model, **kwargs)


def run_absolute_jury(
    conversations: list[str],
    references: list[str],
    predictions: list[str],
    run_name: str,
    output_dir: pathlib.Path,
    judge_type: str,
) -> None:
    if judge_type in LARGE_JUDGE_MODEL_PATHS:
        run_large_model_absolute_jury(
            conversations=conversations,
            references=references,
            predictions=predictions,
            run_name=run_name,
            output_dir=output_dir,
            judge_type=judge_type,
        )
        return

    import utils.prometheus as vendor_prometheus
    from utils import constants as vendor_constants

    print(f"Loading Prometheus absolute judge from {PROMETHEUS_MODEL_PATH} ...")
    judge = _load_prometheus_judge("absolute")

    for aspect, rubric_data in vendor_constants.prometheus_absolute_rubric_data.items():
        print(f"  Scoring aspect: {aspect}")
        scores = vendor_prometheus.get_absolute_score(
            absolute_judge=judge,
            conversations_list=conversations,
            reference_list=references,
            model_response_list=predictions,
            prometheus_absolute_rubric_data=rubric_data,
        )
        vendor_prometheus._save_prometheus_absolute_scores(
            prometheus_scores=scores,
            model_A_name=run_name,
            aspect=aspect,
            path=str(output_dir),
        )
    print("Absolute jury complete.")


def run_relative_jury(
    conversations: list[str],
    references: list[str],
    predictions_a: list[str],
    predictions_b: list[str],
    run_name_a: str,
    run_name_b: str,
    output_dir: pathlib.Path,
    judge_type: str,
) -> None:
    if judge_type in LARGE_JUDGE_MODEL_PATHS:
        run_large_model_relative_jury(
            conversations=conversations,
            references=references,
            predictions_a=predictions_a,
            predictions_b=predictions_b,
            run_name_a=run_name_a,
            run_name_b=run_name_b,
            output_dir=output_dir,
            judge_type=judge_type,
        )
        return

    import utils.prometheus as vendor_prometheus

    print(f"Loading Prometheus relative judge for {run_name_a} vs {run_name_b} ...")
    judge = _load_prometheus_judge("relative")
    scores = vendor_prometheus.get_preference_score(
        conversation_list=conversations,
        reference_list=references,
        model_A_response_list=predictions_a,
        model_B_response_list=predictions_b,
        relative_judge_model=judge,
    )
    vendor_prometheus._save_prometheus_relative_scores(
        prometheus_scores=scores,
        model_A_name=run_name_a,
        base_name="promethus_relative_score",
        model_B_name=run_name_b,
        path=str(output_dir),
    )
    print("Relative jury complete.")


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_model_path(run_name: str, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    # run_finetune.sh sets OUTPUT_DIR=…/eval/results/<run-name>/model
    # finetune.py appends /final_model
    return PROJECT_ROOT / "eval" / "results" / run_name / "model" / "final_model"


def get_or_run_inference(
    run_name: str,
    model_path: pathlib.Path,
    examples: list[dict],
    output_dir: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str]]:
    """Return (conversations, references, predictions), using cache if available."""
    cached = load_predictions_cache(output_dir)
    if cached is not None:
        convs = [r["src"] for r in cached]
        refs = [r["tgt"] for r in cached]
        preds = [r["prediction"] for r in cached]
        return convs, refs, preds

    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        print("  Run eval/run_finetune.sh first to train the model.", file=sys.stderr)
        sys.exit(1)

    preds = run_inference(
        model_path=str(model_path),
        examples=examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    save_predictions(output_dir, examples, preds)
    convs = [ex["src"] for ex in examples]
    refs = [ex["tgt"] for ex in examples]
    return convs, refs, preds


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    examples = load_aci_test(args.data_path)
    print(f"Loaded {len(examples)} ACI-bench test examples from {args.data_path}")

    results_root = pathlib.Path(args.results_dir)
    out_a = results_root / args.run_name
    out_a.mkdir(parents=True, exist_ok=True)

    model_path_a = resolve_model_path(args.run_name, args.model_path)
    conversations, references, predictions_a = get_or_run_inference(
        args.run_name, model_path_a, examples, out_a, args
    )

    # ── auto metrics ──────────────────────────────────────────────────────────
    if not args.skip_auto_metrics:
        compute_auto_metrics(predictions_a, references, out_a, args.run_name)

    # ── LLM jury ─────────────────────────────────────────────────────────────
    if not args.skip_llm_judge:
        run_absolute_jury(
            conversations=conversations,
            references=references,
            predictions=predictions_a,
            run_name=args.run_name,
            output_dir=out_a,
            judge_type=args.judge,
        )

        if args.run_name_b:
            out_b = results_root / args.run_name_b
            out_b.mkdir(parents=True, exist_ok=True)
            model_path_b = resolve_model_path(args.run_name_b, args.model_path_b)
            _, _, predictions_b = get_or_run_inference(
                args.run_name_b, model_path_b, examples, out_b, args
            )
            # relative results written alongside model-A output
            run_relative_jury(
                conversations=conversations,
                references=references,
                predictions_a=predictions_a,
                predictions_b=predictions_b,
                run_name_a=args.run_name,
                run_name_b=args.run_name_b,
                output_dir=out_a,
                judge_type=args.judge,
            )

    print("Done.")


if __name__ == "__main__":
    main()
