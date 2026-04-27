"""
Majority-vote aggregation over per-judge pairwise (relative) preference CSVs.

After running run_pairwise_all.sh (which submits 3 judge jobs per pair), this
script reads the 3 per-judge relative CSVs for a given (run_a, run_b) pair and
produces a majority-voted preference per example.

Voting rule:
  - Empty strings / parse failures are excluded.
  - statistics.mode of valid A/B votes.
  - If all three differ or no valid vote, records '' (inconclusive).
  - Tie (e.g. one A, one B, one invalid) resolved by mode of the two valid votes.

Output:
  eval/benchmarking/results/<run_a>/relative_majority_<run_a>_<run_b>_<date>.csv

Prints a win-rate summary table to stdout.

Usage:
    cd eval/benchmarking
    python compute_majority_vote_relative.py --run_a <run_a> --run_b <run_b> [--results_dir ./results]
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import statistics
from datetime import datetime

import pandas as pd


PAIR_CONFIGS = [
    ("finetune-diverse-k100-n1000-gemma3-4b-3ep",  "finetune-full-medsynth-gemma3-4b-3ep"),
    ("finetune-diverse-k100-n2500-gemma3-4b-3ep",  "finetune-full-medsynth-gemma3-4b-3ep"),
    ("finetune-diverse-k100-n5000-gemma3-4b-3ep",  "finetune-full-medsynth-gemma3-4b-3ep"),
    ("finetune-diverse-k100-n7500-gemma3-4b-3ep",  "finetune-full-medsynth-gemma3-4b-3ep"),
    ("finetune-diverse-k100-n1000-llama32-3b-3ep", "finetune-full-medsynth-llama32-3b-3ep"),
    ("finetune-diverse-k100-n2500-llama32-3b-3ep", "finetune-full-medsynth-llama32-3b-3ep"),
    ("finetune-diverse-k100-n5000-llama32-3b-3ep", "finetune-full-medsynth-llama32-3b-3ep"),
    ("finetune-diverse-k100-n7500-llama32-3b-3ep", "finetune-full-medsynth-llama32-3b-3ep"),
    ("finetune-diverse-k100-n1000-qwen25-3b-3ep",  "finetune-full-medsynth-qwen25-3b-3ep"),
    ("finetune-diverse-k100-n2500-qwen25-3b-3ep",  "finetune-full-medsynth-qwen25-3b-3ep"),
    ("finetune-diverse-k100-n5000-qwen25-3b-3ep",  "finetune-full-medsynth-qwen25-3b-3ep"),
    ("finetune-diverse-k100-n7500-qwen25-3b-3ep",  "finetune-full-medsynth-qwen25-3b-3ep"),
]

# (column_suffix, glob_prefix)
JUDGE_PATTERNS = [
    ("prometheus",  "promethus_relative_score"),   # note: existing typo preserved
    ("gemma3_27b",  "relative_gemma3_27b"),
    ("qwen25_32b",  "relative_qwen25_32b"),
]


def find_judge_csv(run_dir: pathlib.Path, prefix: str, run_a: str, run_b: str) -> pathlib.Path | None:
    pattern = str(run_dir / f"{prefix}_{run_a}_{run_b}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return pathlib.Path(sorted(matches)[-1])


def majority_pref(votes: list[str]) -> str:
    valid = [v for v in votes if v in ("A", "B")]
    if not valid:
        return ""
    try:
        return statistics.mode(valid)
    except statistics.StatisticsError:
        return ""


def compute_majority_vote_relative(run_a: str, run_b: str, results_dir: pathlib.Path) -> dict:
    run_dir = results_dir / run_a
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    judge_dfs: dict[str, pd.DataFrame] = {}
    for suffix, prefix in JUDGE_PATTERNS:
        csv_path = find_judge_csv(run_dir, prefix, run_a, run_b)
        if csv_path is None:
            print(f"  [WARN] missing: {prefix}_{run_a}_{run_b}_*.csv — skipping judge {suffix}")
            continue
        df = pd.read_csv(csv_path, sep="|")
        judge_dfs[suffix] = df
        print(f"  Loaded {csv_path.name} ({len(df)} rows)")

    if len(judge_dfs) < 2:
        print(f"  [SKIP] {run_a} vs {run_b}: need at least 2 judges, found {len(judge_dfs)}")
        return {}

    n_rows = min(len(df) for df in judge_dfs.values())
    majority_prefs: list[str] = []
    per_judge_cols: dict[str, list[str]] = {s: [] for s in judge_dfs}

    for idx in range(n_rows):
        votes = []
        for suffix, df in judge_dfs.items():
            raw = str(df.iloc[idx].get("Preference", "")).strip().upper()
            pref = raw if raw in ("A", "B") else ""
            per_judge_cols[suffix].append(pref)
            votes.append(pref)
        majority_prefs.append(majority_pref(votes))

    first_df = next(iter(judge_dfs.values())).iloc[:n_rows].copy()
    for suffix, col in per_judge_cols.items():
        first_df[f"pref_{suffix}"] = col
    first_df["majority_preference"] = majority_prefs

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = run_dir / f"relative_majority_{run_a}_{run_b}_{today}.csv"
    first_df.to_csv(out_path, index=False, sep="|")
    print(f"  → {out_path.name}")

    a_wins = majority_prefs.count("A")
    b_wins = majority_prefs.count("B")
    inconclusive = n_rows - a_wins - b_wins
    return {"run_a": run_a, "run_b": run_b, "A_wins": a_wins, "B_wins": b_wins,
            "inconclusive": inconclusive, "total": n_rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute majority-voted pairwise preference scores")
    p.add_argument("--run_a", default=None,
                   help="Subset-finetuned run name (if omitted, processes all 12 canonical pairs)")
    p.add_argument("--run_b", default=None,
                   help="Full-MedSynth run name (required when --run_a is given)")
    p.add_argument("--results_dir", default=str(pathlib.Path(__file__).parent / "results"),
                   help="Root results directory (default: ./results)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = pathlib.Path(args.results_dir)

    if args.run_a:
        if not args.run_b:
            raise SystemExit("ERROR: --run_b is required when --run_a is specified")
        pairs = [(args.run_a, args.run_b)]
    else:
        pairs = PAIR_CONFIGS

    summaries = []
    for run_a, run_b in pairs:
        print(f"\n--- {run_a} vs {run_b} ---")
        result = compute_majority_vote_relative(run_a, run_b, results_dir)
        if result:
            summaries.append(result)

    if summaries:
        print("\n=== Pairwise majority-vote summary (A = diverse subset, B = full MedSynth) ===")
        print(f"  {'Run A (diverse subset)':<52s}  A-wins  B-wins  inconclusive")
        for s in summaries:
            print(f"  {s['run_a']:<52s}  {s['A_wins']:>6}  {s['B_wins']:>6}  {s['inconclusive']:>12}")


if __name__ == "__main__":
    main()
