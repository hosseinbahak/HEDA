# app/main_app.py
import uuid
from flask import Flask, render_template, request, jsonify
from app.orchestrator import Orchestrator
from app.utils.logging_setup import configure_logging

logger = configure_logging("heda")

app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    trace_id = body.get("trace_id") or str(uuid.uuid4())

    if not text:
        logger.warning("empty_text", extra={"extra": {"trace_id": trace_id}})
        return jsonify({"error": "No text provided"}), 400

    logger.info("request_received", extra={"extra": {"trace_id": trace_id, "len": len(text)}})

    orch = Orchestrator(trace_id=trace_id)
    result = orch.evaluate_text(text)

    logger.info("request_complete", extra={
        "extra": {"trace_id": trace_id, "verdict": result["summary"]["verdict"], "confidence": result["summary"]["confidence"]}
    })
    return jsonify(result), 200

@app.route("/api/health")
def health():
    return jsonify({"ok": True}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
