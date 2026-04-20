"""
MedSynth dataset loader for finetuning.

Loads Note/Dialogue pairs from a local CSV and formats them as Llama3
instruction-tuned prompts suitable for SFTTrainer.

Usage:
    from eval.utils.dataset import load_medsynth
    ds = load_medsynth("/path/to/MedSynth_huggingface_final.csv", num_samples=1000)
"""

from __future__ import annotations

import pandas as pd
from datasets import Dataset

from utils.constants import DIAL2NOTE_SYSTEM_PROMPT


def _format_prompt(dialogue: str, note: str, system_prompt: str) -> str:
    """Format a single Dialogue→Note pair as a Llama3 chat prompt."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"This is the conversation: {dialogue}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{note}<|eot_id|>"
    )


def load_medsynth(
    data_path: str,
    num_samples: int | None = None,
    seed: int = 42,
    system_prompt: str = DIAL2NOTE_SYSTEM_PROMPT,
) -> Dataset:
    """Load MedSynth CSV and return a HuggingFace Dataset with a 'prompt' column.

    Args:
        data_path: Path to MedSynth_huggingface_final.csv (columns: Note, Dialogue, ...).
        num_samples: If set, shuffle and take this many rows. None = use full dataset.
        seed: Random seed for shuffling (only used when num_samples is set).
        system_prompt: System instruction prepended to every prompt.

    Returns:
        datasets.Dataset with a single 'prompt' column of formatted Llama3 prompts.
    """
    if str(data_path).endswith(".jsonl"):
        # JSONL from prismatic-synthesis: fields are "prompt" (dialogue) and "completion" (note)
        df = pd.read_json(data_path, lines=True).rename(columns={"prompt": "Dialogue", "completion": "Note"})
    else:
        df = pd.read_csv(data_path)

    df = df.dropna(subset=["Note", "Dialogue"])
    df = df.drop_duplicates(subset=["Note", "Dialogue"])

    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)

    prompts = [
        _format_prompt(str(row["Dialogue"]), str(row["Note"]), system_prompt)
        for _, row in df.iterrows()
    ]

    return Dataset.from_dict({"prompt": prompts})
