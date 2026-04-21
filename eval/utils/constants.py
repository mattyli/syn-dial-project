"""
Constants and configuration for MedSynth finetuning.

Adapted from vendor/MedSynth/eval/utils/constants.py.
HuggingFace Hub paths removed; model is saved locally.
"""

import torch

DIAL2NOTE_SYSTEM_PROMPT = (
    "You are an assistant for medical professionals, specializing in summarizing their "
    "conversations with patients. Your role is to accurately and comprehensively summarize "
    "these conversations in the SOAP (Subjective, Objective, Assessment, Plan) format. "
    "Ensure that each summary is thorough and precise, reflecting all relevant details from "
    "the conversation to provide a reliable medical record."
)

DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"

TUNING_CONFIG = {
    "model_config": {
        "max_seq_length": 5120,
        "dtype": torch.bfloat16,
        "load_in_4bit": True,
    },
    "lora_config": {
        "r": 16,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": True,
        "use_rslora": False,
        "use_dora": False,
        "loftq_config": None,
    },
    "training_config": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "num_train_epochs": 4,
        "learning_rate": 2e-4,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 42,
        "dataset_text_field": "prompt",
        "max_length": 5120,   # renamed from max_seq_length in trl >= 0.10
        "dataset_num_proc": 2,
        "packing": False,
    },
}
