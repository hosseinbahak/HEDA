# eval/run_comprehensive_evaluation.py
# Drop-in replacement

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent  # project root
print(f"[paths] THIS_DIR={THIS_DIR}")
print(f"[paths] looking for runner at {THIS_DIR / 'runners' / 'run_heda.py'}")

PYTHON = sys.executable or "python3"

def make_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def system_to_cmd(system_name: str, dataset: Path, out_file: Path,
                  max_rounds: int, consensus_threshold: float, confidence_threshold: float) -> List[str]:
    """
    Map a system name to a concrete command.
    We reuse the existing demo runner: eval/runners/run_heda.py
    """
    runner = (THIS_DIR / "runners" / "run_heda.py").resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found at {runner}")

    system = system_name.strip().lower()
    if system in ("heda-roundtable", "heda", "heda-rt", "heda_judge"):
        return [
            PYTHON, str(runner),
            "--input", str(dataset),
            "--out", str(out_file),
            "--mode", "roundtable",
            "--max_rounds", str(max_rounds),
            "--consensus_threshold", str(consensus_threshold),
            "--confidence_threshold", str(confidence_threshold),
        ]
    elif system in ("heda-traditional", "heda-single", "heda-base"):
        return [
            PYTHON, str(runner),
            "--input", str(dataset),
            "--out", str(out_file),
            "--mode", "traditional",
            "--max_rounds", "1",  # traditional path ignores extra rounds in demo
            "--consensus_threshold", str(consensus_threshold),
            "--confidence_threshold", str(confidence_threshold),
        ]
    else:
        raise ValueError(f"Unknown system: {system_name}")

async def run_cmd(cmd: List[str], cwd: Path = REPO_ROOT) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    out_text = stdout.decode("utf-8", errors="ignore")
    print(f"\n$ {' '.join(cmd)}\n{out_text}")
    return proc.returncode

async def run_system_and_metrics(system: str,
                                 dataset: Path,
                                 out_dir: Path,
                                 gold_path: Path,
                                 max_rounds: int,
                                 consensus_threshold: float,
                                 confidence_threshold: float) -> None:
    sys_out_dir = out_dir / system.replace(" ", "_")
    make_dirs(sys_out_dir)

    # 1) Predictions
    pred_path = sys_out_dir / "predictions.jsonl"
    cmd_pred = system_to_cmd(system, dataset, pred_path, max_rounds, consensus_threshold, confidence_threshold)
    rc = await run_cmd(cmd_pred)
    if rc != 0:
        raise RuntimeError(f"Prediction run failed for {system} (rc={rc})")

    # 2) Basic metrics
    metrics_path = sys_out_dir / "metrics.json"
    metrics_script = (THIS_DIR / "metrics" / "compute_metrics.py").resolve()
    print(f"[paths] looking for metrics at {metrics_script}")
    if not metrics_script.exists():
        raise FileNotFoundError(f"Metrics script not found: {metrics_script}")


    cmd_metrics = [
        PYTHON, str(metrics_script),
        "--gold", str(gold_path),
        "--pred", str(pred_path),
        "--report", str(metrics_path),
    ]
    rc = await run_cmd(cmd_metrics)
    if rc != 0:
        raise RuntimeError(f"Metrics run failed for {system} (rc={rc})")

    # 3) Cost/latency placeholder (the demo runner doesn’t log costs; create an empty stub)
    cost_latency_path = sys_out_dir / "cost_latency.json"
    if not cost_latency_path.exists():
        cost_stub = {"tokens_in": None, "tokens_out": None, "dollars": None, "latency_s": None, "note": "Populate from your LLM client wrappers."}
        cost_latency_path.write_text(json.dumps(cost_stub, ensure_ascii=False, indent=2))

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive evaluation driver for HEDA and baselines.")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL (HEDA schema).")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save results.")
    p.add_argument("--systems", nargs="+", default=["HEDA-RoundTable", "HEDA-Traditional"],
                   help="Systems to compare. Known: HEDA-RoundTable, HEDA-Traditional")
    p.add_argument("--parallel", action="store_true", help="Run systems in parallel.")
    p.add_argument("--max-concurrency", type=int, default=2, help="Max concurrent jobs when --parallel is set.")
    # Compatibility flag (alias)
    p.add_argument("--use-heda-judge", action="store_true",
                   help="Alias: append HEDA-RoundTable to --systems if not already present.")
    # HEDA knobs
    p.add_argument("--max_rounds", type=int, default=2, help="Rounds for round-table mode.")
    p.add_argument("--consensus_threshold", type=float, default=0.7, help="Consensus threshold.")
    p.add_argument("--confidence_threshold", type=float, default=0.6, help="Confidence threshold.")
    return p.parse_args()

async def main_async():
    args = parse_args()

    dataset = Path(args.dataset).resolve()
    out_dir = Path(args.output_dir).resolve()
    make_dirs(out_dir)

    systems = [s for s in args.systems]
    if args.use_heda_judge and "HEDA-RoundTable" not in [s.lower() for s in systems]:
        systems.append("HEDA-RoundTable")

    # sanity
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    # gold is the same dataset for detection metrics (expects gold_has_error)
    gold_path = dataset

    tasks = []
    sem = asyncio.Semaphore(args.max_concurrency)

    async def runner(sysname: str):
        async with sem:
            await run_system_and_metrics(
                sysname, dataset, out_dir, gold_path,
                args.max_rounds, args.consensus_threshold, args.confidence_threshold
            )

    if args.parallel:
        for s in systems:
            tasks.append(asyncio.create_task(runner(s)))
        await asyncio.gather(*tasks)
    else:
        for s in systems:
            await runner(s)

    # summary CSV (simple merge)
    summary_csv = out_dir / "results_summary.csv"
    try:
        import csv
        rows = []
        for s in systems:
            metrics_file = out_dir / s.replace(" ", "_") / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                m["system"] = s
                rows.append(m)
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            with open(summary_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            print(f"[OK] Wrote summary CSV → {summary_csv}")
    except Exception as e:
        print(f"[WARN] Could not write summary CSV: {e}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)

if __name__ == "__main__":
    main()
