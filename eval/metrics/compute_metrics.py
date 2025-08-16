# eval/metrics/compute_metrics.py
import json, argparse, math, sys
from collections import defaultdict
import numpy as np

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def bin_confidence_ece(y_true, y_prob, bins=10):
    # Expected Calibration Error for binary classification
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0,1,bins+1)
    ece = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        idx = (y_prob >= lo) & (y_prob < hi if i < bins-1 else y_prob <= hi)
        if not np.any(idx): continue
        bin_acc = np.mean(y_true[idx] == (y_prob[idx] >= 0.5))
        bin_conf = np.mean(y_prob[idx])
        w = np.mean(idx)
        ece += w * abs(bin_acc - bin_conf)
    return float(ece)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="JSONL with fields: id, gold_has_error")
    ap.add_argument("--pred", required=True, help="JSONL normalized output")
    ap.add_argument("--report", required=True, help="Where to write metrics JSON")
    args = ap.parse_args()

    gold = {ex["id"]: bool(ex.get("gold_has_error")) for ex in load_jsonl(args.gold)}
    pred = load_jsonl(args.pred)

    y_true, y_pred, y_prob = [], [], []
    for ex in pred:
        sid = ex["id"]
        if sid not in gold: continue
        y_true.append(gold[sid])
        pr = ex["prediction"]
        y_pred.append(bool(pr.get("has_error")))
        y_prob.append(float(pr.get("confidence", 0.5)))

    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    y_prob = np.array(y_prob, dtype=float)

    acc = float(np.mean(y_true == y_pred))
    prec = float(np.sum(y_true & y_pred) / max(np.sum(y_pred), 1))
    rec = float(np.sum(y_true & y_pred) / max(np.sum(y_true), 1))
    f1 = 0.0 if (prec+rec)==0 else (2*prec*rec)/(prec+rec)
    ece = bin_confidence_ece(y_true, y_prob)

    metrics = {
        "samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "precision_error": round(prec, 4),
        "recall_error": round(rec, 4),
        "f1_error": round(f1, 4),
        "ece": round(ece, 4)
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
