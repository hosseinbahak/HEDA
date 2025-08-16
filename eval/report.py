# eval/report.py
import argparse, json, os, numpy as np

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def load_gold(path):
    return {ex["id"]: bool(ex.get("gold_has_error")) for ex in load_jsonl(path)}

def summarize_system(gold, system_path):
    exs = load_jsonl(system_path)
    y_true, y_pred = [], []
    for ex in exs:
        sid = ex["id"]
        if sid not in gold: continue
        y_true.append(gold[sid])
        y_pred.append(bool(ex["prediction"]["has_error"]))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    name = os.path.basename(system_path)
    return name, acc, len(y_true)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="One or more normalized JSONL result files")
    ap.add_argument("--out", required=True, help="HTML path")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    rows = []
    for sp in args.systems:
        name, acc, n = summarize_system(gold, sp)
        rows.append((name, acc, n))

    rows.sort(key=lambda x: x[1], reverse=True)

    html = ["<html><head><meta charset='utf-8'><title>HEDA Eval Report</title>"]
    html.append("<style>body{font-family:system-ui,Segoe UI,Roboto,Arial;max-width:900px;margin:2rem auto;padding:0 1rem;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;} th{background:#f7f7f7;}</style>")
    html.append("</head><body><h1>Evaluation Report</h1>")
    html.append("<p>Normalized comparison across systems (higher accuracy is better).</p>")
    html.append("<table><tr><th>System</th><th>Accuracy</th><th>Samples</th></tr>")
    for name, acc, n in rows:
        html.append(f"<tr><td>{name}</td><td>{acc:.3f}</td><td>{n}</td></tr>")
    html.append("</table></body></html>")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(html))
    print(f"Wrote report: {args.out}")

if __name__ == "__main__":
    main()
