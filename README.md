# HEDA Framework (Lite) — Multi‑Agent Reasoning Judge + Evaluation Harness

This is a **self‑contained**, runnable implementation of a streamlined HEDA framework
with a **web UI**, **multi‑agent orchestrator** (Prosecutor → Reflector → Defense → Judge),
a **mock LLM backend** (works offline), **optional OpenRouter support**, and a full
**evaluation harness** to compare HEDA against other frameworks using metrics inspired
by CheckEval, JudgeLM, JudgeBench, and Multi‑Agent Judge.

## Quick Start (offline demo)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Run the web app
python app/main_app.py
# open http://127.0.0.1:5000 and paste some reasoning text to evaluate

# 3) Batch evaluate on the toy dataset
python eval/runners/run_heda.py --input eval/datasets/toy_reasoning.jsonl --out results/heda_toy.jsonl

# 4) Compute metrics (requires the ground-truth labels included in the toy dataset)
python eval/metrics/compute_metrics.py \
  --gold eval/datasets/toy_reasoning.jsonl \
  --pred results/heda_toy.jsonl \
  --report results/heda_metrics.json

# 5) Make a compact HTML report aggregating one or more frameworks
python eval/report.py \
  --gold eval/datasets/toy_reasoning.jsonl \
  --systems results/heda_toy.jsonl \
  --out results/report.html
