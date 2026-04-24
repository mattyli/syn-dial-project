# bench — Run the ACI-bench evaluation suite

Submit one or more benchmarking Slurm jobs for finetuned Dialogue→Note models.

## What this command does

1. Lists all available finetuned runs in `eval/results/` (those that have a `model/final_model/` adapter).
2. Determines which run(s) to benchmark based on `$ARGUMENTS` (see below).
3. Submits a `run_benchmarking.sh` Slurm job for each selected run.
4. Reports the Slurm job ID(s) and the path(s) where results will appear.

## Argument parsing

`$ARGUMENTS` is the raw text the user typed after `/bench`.  Parse it as follows:

- **Empty / "all"**: benchmark every run that has `eval/results/<name>/model/final_model/`.
- **Single run name** (e.g. `finetune-diverse-k100-n1000`): benchmark just that run.
- **Two run names separated by "vs"** (e.g. `finetune-diverse-k100-n1000 vs finetune-diverse-k100-n2500`): benchmark the first run and also submit a pairwise relative-jury job comparing the two (`--run_name_b`).
- **`--skip-llm`** anywhere in the args: pass `--skip_llm` to the script (skip Prometheus judge, auto-metrics only).
- **`--skip-auto`** anywhere in the args: pass `--skip_auto` to the script.

## Steps to execute

1. Run `ls /project/6101844/mattli/syn-dial-project/eval/results/` to enumerate available runs.
2. Filter to runs that have a `model/final_model/adapter_config.json` file.
3. For each target run, call:
   ```bash
   cd /project/6101844/mattli/syn-dial-project/eval/benchmarking
   bash run_benchmarking.sh --run_name <NAME> [--run_name_b <NAME_B>] [--skip_auto] [--skip_llm]
   ```
4. Report back: run name(s), Slurm job ID(s), and result paths (`eval/benchmarking/results/<name>/`). Tell the user they can monitor with `squeue -j <job_id>`.

## Output locations

- Logs: `eval/benchmarking/results/<run_name>/slurm.out|err`
- Auto metrics: `eval/benchmarking/results/<run_name>/auto_metrics.json`
- Prometheus aspect CSVs: `eval/benchmarking/results/<run_name>/<aspect>_Absolute_prometheus_scores_*.csv`
- Pairwise comparison: `eval/benchmarking/results/<run_name>/promethus_relative_score_*.csv`
