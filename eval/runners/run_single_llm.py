#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a single LLM baseline over a JSONL dataset and emit a JSONL result file.
We keep this minimal so the comprehensive runner can call it for multiple "systems".
"""

import argparse, json, sys
from pathlib import Path
from datetime import datetime

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--out", required=True, help="Path to output JSONL")
    ap.add_argument("--model", required=True, help="Model/system name")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy baseline: echoes each example with a trivial "analysis".
    with out_path.open("w", encoding="utf-8") as w:
        for ex in read_jsonl(in_path):
            ex_id = ex.get("id") or ex.get("example_id") or None
            q = ex.get("question") or ex.get("prompt") or ""
            pred = {
                "id": ex_id,
                "system": args.model,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "final_verdict": "Unknown",
                "confidence": 0.5,
                "reasoning": f"Baseline runner for {args.model} did not judge; stub output.",
                "meta": {"question_preview": q[:120]}
            }
            w.write(json.dumps(pred, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
