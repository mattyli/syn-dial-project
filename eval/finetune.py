"""
QLoRA finetuning of Llama models on MedSynth Dialogue→Note data.

Adapted from vendor/MedSynth/eval/utils/model_tuner.py.
Saves the LoRA adapter locally instead of pushing to HuggingFace Hub.

Usage:
    python finetune.py \\
        --data_path /project/aip-zhu2048/mattli/syn-dial-project/MedSynth_huggingface_final.csv \\
        --output_dir /project/aip-zhu2048/mattli/syn-dial-project/eval/results/my-run/model \\
        [--num_samples 5000] \\
        [--base_model unsloth/Meta-Llama-3.1-8B-Instruct] \\
        [--epochs 4] \\
        [--lora_r 16] \\
        [--seed 42]
"""

import unsloth  # noqa: F401 — must be first import before trl/transformers/peft

import argparse
import copy
import pathlib

import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from utils.constants import DEFAULT_BASE_MODEL, TUNING_CONFIG
from utils.dataset import load_medsynth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA finetune Llama on MedSynth data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/project/aip-zhu2048/mattli/syn-dial-project/MedSynth_huggingface_final.csv",
        help="Path to MedSynth_huggingface_final.csv",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of training samples (None = full dataset)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model name or path (unsloth-compatible)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained LoRA adapter and tokenizer",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(TUNING_CONFIG)
    cfg["lora_config"]["r"] = args.lora_r
    cfg["training_config"]["num_train_epochs"] = args.epochs
    cfg["training_config"]["seed"] = args.seed

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.data_path} (num_samples={args.num_samples}) ...")
    dataset = load_medsynth(args.data_path, num_samples=args.num_samples, seed=args.seed)
    print(f"  {len(dataset)} training examples")

    # ── 2. Model + tokenizer ──────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=cfg["model_config"]["max_seq_length"],
        dtype=cfg["model_config"]["dtype"],
        load_in_4bit=cfg["model_config"]["load_in_4bit"],
    )

    # ── 3. LoRA / PEFT ───────────────────────────────────────────────────────
    lora = cfg["lora_config"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing=lora["use_gradient_checkpointing"],
        random_state=args.seed,
        use_rslora=lora["use_rslora"],
        use_dora=lora["use_dora"],
        loftq_config=lora["loftq_config"],
    )

    # ── 4. SFTTrainer (trl ≥ 0.9 uses SFTConfig) ─────────────────────────────
    tc = cfg["training_config"]
    sft_cfg = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        warmup_steps=tc["warmup_steps"],
        num_train_epochs=tc["num_train_epochs"],
        learning_rate=tc["learning_rate"],
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        logging_steps=tc["logging_steps"],
        optim=tc["optim"],
        weight_decay=tc["weight_decay"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        seed=tc["seed"],
        dataset_text_field=tc["dataset_text_field"],
        max_length=tc["max_length"],
        dataset_num_proc=tc["dataset_num_proc"],
        packing=tc["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_cfg,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("Starting training ...")
    train_result = trainer.train()
    print(f"Training complete. Metrics: {train_result.metrics}")

    # ── 6. Save adapter + tokenizer ───────────────────────────────────────────
    final_model_dir = output_dir / "final_model"
    print(f"Saving adapter to {final_model_dir} ...")
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print("Done.")


if __name__ == "__main__":
    main()
