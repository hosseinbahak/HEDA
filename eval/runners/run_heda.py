# eval/runners/run_heda.py
import argparse, json, os, sys, time, signal
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from app.orchestrator import Orchestrator

DEFAULT_TIMEOUT_SECS = int(os.getenv("RUNNER_SAMPLE_TIMEOUT", "90"))

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[reader] JSON error on line {i}: {e}", file=sys.stderr)
                continue
            rid = rec.get("id", f"line_{i}")
            print(f"[reader] loaded #{i}: id={rid}")
            yield rec

def normalize(record: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    # Expect orchestrator returns { summary:{ has_error, verdict, confidence, ... }, ... }
    s = result.get("summary", {})
    return {
        "id": record.get("id"),
        "prediction": {
            "has_error": bool(s.get("has_error", False)),
            "verdict": s.get("verdict", "No Significant Errors"),
            "confidence": float(s.get("confidence", 0.0)),
            "charges": (result.get("round", {})
                        .get("prosecutor", {})
                        .get("content", {})
                        .get("case_file", [])) or []
        },
        "meta": {
            "framework": "HEDA",
            "version": os.getenv("HEDA_VERSION", "runner-1.0"),
        }
    }

def evaluate_one(orchestrator: Orchestrator, prompt: str) -> Dict[str, Any]:
    return orchestrator.evaluate_text(prompt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL with {id, prompt, gold_has_error}")
    ap.add_argument("--out", required=True, help="Path to write predictions JSONL")
    args = ap.parse_args()

    ensure_dir(args.out)

    # Make sure we don’t accidentally hit OpenRouter if you didn’t mean to.
    # Set USE_OPENROUTER=true in .env to use live models.
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    if not use_openrouter:
        os.environ["FALLBACK_TO_MOCK_ON_ERROR"] = "true"

    # One orchestrator reused across items (faster)
    orch = Orchestrator()

    total = 0
    ok = 0
    failed = 0
    start_all = time.time()

    def sigint_handler(signum, frame):
        print("\n[runner] Interrupted. Partial results are in:", args.out, file=sys.stderr)
        sys.exit(130)
    signal.signal(signal.SIGINT, sigint_handler)

    with open(args.out, "a", encoding="utf-8") as fw, ThreadPoolExecutor(max_workers=1) as pool:
        for rec in load_jsonl(args.input):
            total += 1
            rid = rec.get("id", f"sample_{total}")
            prompt = rec.get("prompt", "")
            print(f"[{total}] {rid} …", end="", flush=True)

            fut = pool.submit(evaluate_one, orch, prompt)
            try:
                result = fut.result(timeout=DEFAULT_TIMEOUT_SECS)
                pred = normalize(rec, result)
                fw.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fw.flush()
                os.fsync(fw.fileno())
                ok += 1
                print(" done")
            except FuturesTimeout:
                failed += 1
                fut.cancel()
                err = {"id": rid, "error": f"timeout_{DEFAULT_TIMEOUT_SECS}s"}
                fw.write(json.dumps({"id": rid, "prediction": {"has_error": False, "verdict": "Timeout", "confidence": 0.0, "charges": []}, "meta": {"framework": "HEDA", "version": "runner-1.0", "error": err["error"]}}) + "\n")
                fw.flush()
                os.fsync(fw.fileno())
                print(f" TIMEOUT ({DEFAULT_TIMEOUT_SECS}s)")
            except Exception as e:
                failed += 1
                fw.write(json.dumps({"id": rid, "prediction": {"has_error": False, "verdict": "RunnerException", "confidence": 0.0, "charges": []}, "meta": {"framework": "HEDA", "version": "runner-1.0", "error": str(e)}}) + "\n")
                fw.flush()
                os.fsync(fw.fileno())
                print(f" ERROR: {e}")

    dur = time.time() - start_all
    print(f"\n[runner] finished: total={total} ok={ok} failed={failed} time={dur:.1f}s")
    print(f"[runner] wrote: {args.out}")

if __name__ == "__main__":
    main()
