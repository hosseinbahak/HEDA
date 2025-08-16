#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HEDA round-table runner.
Reads a JSONL dataset and writes a JSONL of HEDA-style verdicts via the Orchestrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# --- Make sure project root is on sys.path ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # .../HEDA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.orchestrator import Orchestrator  # noqa: E402


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
    ap.add_argument("--max_rounds", type=int, default=1)
    ap.add_argument("--consensus_threshold", type=float, default=0.7)
    ap.add_argument("--confidence_threshold", type=float, default=0.6)
    args = ap.parse_args()

    in_path = (PROJECT_ROOT / args.input) if not args.input.startswith(str(PROJECT_ROOT)) else Path(args.input)
    out_path = (PROJECT_ROOT / args.out) if not args.out.startswith(str(PROJECT_ROOT)) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(
        max_rounds=args.max_rounds,
        consensus_threshold=args.consensus_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    with out_path.open("w", encoding="utf-8") as w:
        for ex in read_jsonl(in_path):
            result = orch.evaluate_example(ex)
            result["timestamp"] = datetime.utcnow().isoformat() + "Z"
            result["system"] = "HEDA-RoundTable"
            w.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ HEDA wrote results to {out_path}")

if __name__ == "__main__":
    main()
