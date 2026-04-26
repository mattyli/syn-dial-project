"""
Majority-vote aggregation over per-judge absolute scoring CSVs.

After running run_benchmarking.py with --judge prometheus, --judge gemma3_27b,
and --judge qwen25_32b for the same run, this script reads the per-aspect CSVs
from each judge and produces a majority-voted aggregated CSV per aspect.

Voting rule:
  - Scores of -1 (parse failures) are excluded.
  - statistics.mode of the valid integer scores (1–5).
  - If all three differ (no unique mode), falls back to round(statistics.median(...)).
  - If all judges failed for a row, records -1.

Output:
  eval/benchmarking/results/<run_name>/{aspect}_Absolute_majority_scores_<run_name>_<date>.csv

Usage:
    cd eval/benchmarking
    python compute_majority_vote.py --run_name <run_name> [--results_dir ./results]
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import statistics
from datetime import datetime

import pandas as pd


JUDGE_BASE_NAMES = [
    "Absolute_prometheus_scores",
    "Absolute_gemma3_27b_scores",
    "Absolute_qwen25_32b_scores",
]

ASPECTS = [
    "Hallucination",
    "Critical Omissions",
    "Redundancy",
    "Professional Tone",
    "The logical structure of the note and sentences",
    "Adherence to SOAP format",
    "Section Relevance",
]


def majority_vote(scores: list[int]) -> int:
    valid = [s for s in scores if s != -1]
    if not valid:
        return -1
    try:
        return statistics.mode(valid)
    except statistics.StatisticsError:
        return round(statistics.median(valid))


def find_judge_csv(run_dir: pathlib.Path, aspect: str, base_name: str, run_name: str) -> pathlib.Path | None:
    pattern = str(run_dir / f"{aspect}_{base_name}_{run_name}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return pathlib.Path(sorted(matches)[-1])


def compute_majority_vote(run_name: str, results_dir: pathlib.Path) -> None:
    run_dir = results_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    today = datetime.now().strftime("%Y-%m-%d")
    summary: dict[str, float] = {}

    for aspect in ASPECTS:
        judge_dfs: dict[str, pd.DataFrame] = {}

        for base_name in JUDGE_BASE_NAMES:
            csv_path = find_judge_csv(run_dir, aspect, base_name, run_name)
            if csv_path is None:
                print(f"  [WARN] missing: {aspect} / {base_name} — skipping this judge")
                continue
            df = pd.read_csv(csv_path, sep="|")
            judge_dfs[base_name] = df
            print(f"  Loaded {csv_path.name} ({len(df)} rows)")

        if len(judge_dfs) < 2:
            print(f"  [SKIP] {aspect}: need at least 2 judges, found {len(judge_dfs)}")
            continue

        # Align all DataFrames by row index (all use same 40 ACI-bench examples in order)
        n_rows = min(len(df) for df in judge_dfs.values())
        voted_scores: list[int] = []
        judge_score_cols: dict[str, list[int]] = {b: [] for b in judge_dfs}

        for idx in range(n_rows):
            per_judge = []
            for base_name, df in judge_dfs.items():
                raw = df.iloc[idx]["Score"]
                try:
                    score = int(raw)
                except (ValueError, TypeError):
                    score = -1
                per_judge.append(score)
                judge_score_cols[base_name].append(score)
            voted_scores.append(majority_vote(per_judge))

        # Build output DataFrame from the first available judge (for metadata columns)
        first_df = next(iter(judge_dfs.values())).iloc[:n_rows].copy()
        first_df["Score"] = voted_scores
        # Add individual judge scores for auditability
        for base_name, col_scores in judge_score_cols.items():
            short = base_name.replace("Absolute_", "").replace("_scores", "")
            first_df[f"score_{short}"] = col_scores

        out_path = run_dir / f"{aspect}_Absolute_majority_scores_{run_name}_{today}.csv"
        first_df.to_csv(out_path, index=False, sep="|")
        print(f"  → {out_path.name}")

        valid_scores = [s for s in voted_scores if s != -1]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
        summary[aspect] = mean_score

    print("\n=== Majority vote summary (mean score across 40 examples) ===")
    for aspect, mean in summary.items():
        print(f"  {aspect:<50s}  {mean:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute majority-voted LLM judge scores")
    p.add_argument("--run_name", required=True,
                   help="Name matching eval/benchmarking/results/<run_name>/")
    p.add_argument("--results_dir",
                   default=str(pathlib.Path(__file__).parent / "results"),
                   help="Root results directory (default: ./results)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_majority_vote(
        run_name=args.run_name,
        results_dir=pathlib.Path(args.results_dir),
    )
