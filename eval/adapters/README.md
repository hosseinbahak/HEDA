# Adapters: Converting Other Frameworks to the Normalized Schema

Drop small scripts here that transform outputs from other frameworks (CheckEval,
JudgeLM, JudgeBench, Multi‑Agent Judge) into the normalized JSONL schema used by
this repo.

Template (fill in the parser for your framework's raw outputs):

```python
# eval/adapters/your_framework_adapter.py
import json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to framework-native output")
    ap.add_argument("--out", required=True, help="Normalized JSONL path")
    args = ap.parse_args()

    # TODO: parse your framework's output structure
    # For each sample, construct a record like:
    # {
    #   "id": "<sample-id>",
    #   "prediction": {
    #       "has_error": true/false,
    #       "verdict": "Errors Found" | "No Significant Errors",
    #       "confidence": 0.82
    #   },
    #   "meta": {"framework":"CheckEval"}
    # }

    # Example passthrough (replace with real parsing):
    items = json.load(open(args.input, "r", encoding="utf-8"))
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            norm = {
                "id": it["id"],
                "prediction": {
                    "has_error": bool(it.get("has_error", False)),
                    "verdict": "Errors Found" if it.get("has_error") else "No Significant Errors",
                    "confidence": float(it.get("confidence", 0.5))
                },
                "meta": {"framework":"YourFramework"}
            }
            f.write(json.dumps(norm) + "\n")

if __name__ == "__main__":
    main()
