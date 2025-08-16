import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# --- make repo root importable ---
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from app.orchestrator import Orchestrator
import yaml

# ---------- Env & Logging ----------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("heda.runner")

# ---------- utils ----------
def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(obj: Dict[str, Any], p: Path):
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def build_effective_config(base_cfg_path: Path,
                           max_rounds: Optional[int]) -> Path:
    """
    A small overlay: read base config and optionally override keys like max_rounds.
    Returns path to a temp YAML that Orchestrator will read.
    """
    cfg = load_yaml(base_cfg_path)
    if max_rounds is not None:
        cfg["max_rounds"] = int(max_rounds)
    # می‌توانی در آینده کلیدهای دیگری هم اینجا override کنی (مثلاً roles، thresholds، ...)
    tmp = Path(tempfile.mkdtemp(prefix="heda_cfg_")) / "config.yaml"
    dump_yaml(cfg, tmp)
    return tmp

def get_text(sample: Dict[str, Any]) -> str:
    for k in ("prompt", "question", "problem", "response", "solution"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError("No input text in sample (expected: prompt/question/problem/response/solution).")

def summary_to_prediction(summary: Dict[str, Any]) -> Dict[str, Any]:
    verdict = summary.get("verdict") or summary.get("final_verdict") or ""
    has_error = summary.get("has_error")
    if has_error is None:
        # fallback: استنتاج از verdict
        has_error = str(verdict).strip().lower().startswith("error")
    try:
        conf = float(summary.get("confidence", 0.0))
    except Exception:
        conf = 0.0

    # توضیح: اگر reasoning/ explanation نبود، یک خلاصه‌ی کوتاه بساز
    explanation = (
        summary.get("verdict_reasoning")
        or summary.get("reasoning")
        or summary.get("explanation")
        or f"Verdict={verdict}; consensus={summary.get('consensus')}"
    )
    return {"has_error": bool(has_error), "confidence": conf, "explanation": explanation}

def run_with_orchestrator(sample: Dict[str, Any],
                          config_path: Path,
                          use_roundtable: bool,
                          trace_prefix: str) -> Dict[str, Any]:
    sid = sample.get("id", "unknown")
    text = get_text(sample)

    orch = Orchestrator(
        config_path=str(config_path),
        trace_id=f"{trace_prefix}_{sid}",
        use_roundtable=use_roundtable
    )
    full = orch.evaluate_text(text)

    summary = full.get("summary") if isinstance(full, dict) else None
    if not isinstance(summary, dict):
        logger.warning("No 'summary' in orchestrator output; returning minimal prediction.")
        return {
            "id": sid,
            "prediction": {"has_error": True, "confidence": 0.0, "explanation": "No summary in output"},
            "meta": {"framework": "HEDA-RoundTable" if use_roundtable else "HEDA-Traditional"}
        }

    pred = summary_to_prediction(summary)
    return {
        "id": sid,
        "prediction": pred,
        "meta": {
            "framework": "HEDA-RoundTable" if use_roundtable else "HEDA-Traditional",
            "consensus": summary.get("consensus"),
            "num_charges": summary.get("num_charges"),
        }
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Run HEDA (real Orchestrator; config-driven).")
    ap.add_argument("--input", required=True, help="Input dataset JSONL (HEDA schema).")
    ap.add_argument("--out", required=True, help="Output results JSONL.")
    ap.add_argument("--mode", choices=["roundtable", "traditional"], default="roundtable",
                    help="Use roundtable (LangGraph) or traditional pipeline.")
    ap.add_argument("--config", default="app/config/config.yaml", help="Path to base config.yaml.")
    # این فلگ‌ها دیگر مستقیم به Orchestrator نمی‌روند؛ از طریق overlay در config اعمال می‌شوند:
    ap.add_argument("--max_rounds", type=int, default=None, help="Override config.max_rounds for roundtable.")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    base_cfg = Path(args.config).resolve()

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise SystemExit(1)
    if not base_cfg.exists():
        logger.error(f"Config file not found: {base_cfg}")
        raise SystemExit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Overlay config for runtime overrides (e.g., max_rounds)
    effective_cfg = build_effective_config(base_cfg, args.max_rounds)

    use_rt = (args.mode == "roundtable")
    prefix = "rt" if use_rt else "tr"

    processed = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                continue

            logger.info(f"Processing sample ID: {sample.get('id','unknown')}, mode: {args.mode}")
            try:
                result = run_with_orchestrator(
                    sample=sample,
                    config_path=effective_cfg,
                    use_roundtable=use_rt,
                    trace_prefix=prefix
                )
            except Exception as e:
                logger.exception(f"Evaluation failed: {e}")
                result = {
                    "id": sample.get("id","unknown"),
                    "prediction": {"has_error": True, "confidence": 0.0, "explanation": f"orchestrator error: {e}"},
                    "meta": {"framework": "HEDA-RoundTable" if use_rt else "HEDA-Traditional"}
                }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed += 1

    logger.info(f"Evaluation completed → {out_path} | processed={processed}")

if __name__ == "__main__":
    main()
