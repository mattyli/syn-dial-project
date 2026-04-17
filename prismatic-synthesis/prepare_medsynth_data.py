"""
Convert MedSynth_huggingface_final.csv to JSONL files for G-Vendi gradient collection.

Usage:
    python prismatic-synthesis/prepare_medsynth_data.py

Outputs (relative to project root):
    prismatic-synthesis/g-vendi/data/datasets/medsynth_notes.jsonl
    prismatic-synthesis/g-vendi/data/datasets/medsynth_dialogues.jsonl
    prismatic-synthesis/g-vendi/data/datasets/medsynth_selection.jsonl

Prompt framing (task-conditioned):
    Notes:     prompt=dialogue, completion=note   (dialogue→note direction)
    Dialogues: prompt=note,     completion=dialogue (note→dialogue direction)
    Selection: prompt=normalized_dialogue, completion=note (for gradient-based subset selection)
"""

import json
import re
from pathlib import Path

import pandas as pd


def normalize_speakers(text: str) -> str:
    """Replace raw [doctor] / [patient] markers with canonical Doctor: / Patient: form."""
    text = re.sub(r'\[doctor\]:?\s*', 'Doctor: ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[patient\]:?\s*', 'Patient: ', text, flags=re.IGNORECASE)
    return text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "MedSynth_huggingface_final.csv"
OUT_DIR = Path(__file__).resolve().parent / "g-vendi" / "data" / "datasets"

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# The Note column has a leading space
note_col = " Note"
dialogue_col = "Dialogue"

assert note_col in df.columns, f"Column '{note_col}' not found; columns: {df.columns.tolist()}"
assert dialogue_col in df.columns, f"Column '{dialogue_col}' not found; columns: {df.columns.tolist()}"

notes_path = OUT_DIR / "medsynth_notes.jsonl"
dialogues_path = OUT_DIR / "medsynth_dialogues.jsonl"
selection_path = OUT_DIR / "medsynth_selection.jsonl"

with open(notes_path, "w") as f_notes, open(dialogues_path, "w") as f_dialogues, open(selection_path, "w") as f_sel:
    for i, row in df.iterrows():
        note = str(row[note_col]).strip()
        dialogue = str(row[dialogue_col]).strip()

        # Notes: dialogue→note (measure note diversity in task context)
        f_notes.write(json.dumps({"id": f"note_{i}", "prompt": dialogue, "completion": note}) + "\n")

        # Dialogues: note→dialogue (measure dialogue diversity in task context)
        f_dialogues.write(json.dumps({"id": f"dialogue_{i}", "prompt": note, "completion": dialogue}) + "\n")

        # Selection: normalized dialogue→note (for gradient-based diverse subset selection)
        f_sel.write(json.dumps({"id": f"note_{i}", "prompt": normalize_speakers(dialogue), "completion": note}) + "\n")

print(f"Wrote {len(df)} notes to {notes_path}")
print(f"Wrote {len(df)} dialogues to {dialogues_path}")
print(f"Wrote {len(df)} normalized selection examples to {selection_path}")
