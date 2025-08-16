# app/orchestrator.py
import json, yaml, os, time
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List
from app.agents.base import LLMProvider
from app.agents.prosecutor import ProsecutorAgent
from app.agents.reflector import ReflectorAgent
from app.agents.defense import DefenseAgent
from app.agents.judge import JudgeAgent
from app.services.consensus import calculate_consensus
from app.services.bias import detect_biases
from app.utils.logging_setup import configure_logging

logger = configure_logging("heda.orchestrator")

@dataclass
class AgentRun:
    name: str
    model: str
    content: Dict[str, Any]
    confidence: float
    duration_ms: float

class Orchestrator:
    def __init__(self, config_path="app/config/config.yaml", trace_id: Optional[str] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.trace_id = trace_id or os.getenv("TRACE_ID") or "no-trace"
        self.save_artifacts = os.getenv("TRACE_SAVE_ARTIFACTS", "false").lower() == "true"

        self.provider = LLMProvider(use_openrouter=bool(self.config.get("use_openrouter", False)))

        roles_cfg = self.config.get("roles") or {}
        tiers = self.config["model_tiers"]
        self.models = {
            "prosecutor": roles_cfg.get("prosecutor", tiers["balanced"][0]),
            "reflector":  roles_cfg.get("reflector",  tiers["balanced"][0]),
            "defense":    roles_cfg.get("defense",    tiers["balanced"][0]),
            "judge":      roles_cfg.get("judge",      tiers["premium"][0]),
        }

        self.prosecutor = ProsecutorAgent("prosecutor", self.models["prosecutor"], "", self.provider)
        self.reflector  = ReflectorAgent("reflector",  self.models["reflector"],  "", self.provider)
        self.defense    = DefenseAgent("defense",    self.models["defense"],    "", self.provider)
        self.judge      = JudgeAgent("judge",      self.models["judge"],      "", self.provider)

        logger.debug("orchestrator_init", extra={"extra": {"trace_id": self.trace_id, "models": self.models}})

    # ---------- helpers ----------
    def _artifact_dir(self):
        d = os.path.join("results", "traces", self.trace_id)
        if self.save_artifacts:
            os.makedirs(d, exist_ok=True)
        return d

    def _save_json(self, name: str, payload: Dict[str, Any]):
        if not self.save_artifacts: return
        path = os.path.join(self._artifact_dir(), name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _time_call(self, fn, *args, **kwargs):
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        return res, dt

    def _summarize_prosecutor(self, content: Dict[str, Any]) -> str:
        case = content.get("case_file", []) or []
        if not case: return "Filed 0 charges."
        kinds = [c.get("error_code","?") for c in case]
        return f"Filed {len(case)} charge(s): {', '.join(kinds)}"

    def _summarize_reflector(self, content: Dict[str, Any]) -> str:
        cps = content.get("consensus_points", []) or []
        q = content.get("confidence_in_debate_quality")
        qtxt = f", debate_quality={q:.2f}" if isinstance(q,(int,float)) else ""
        return f"Consensus on {len(cps)} charge(s){qtxt}."

    def _summarize_defense(self, content: Dict[str, Any]) -> str:
        r = content.get("rebuttals", {}) or {}
        if not r: return "No rebuttals."
        weak = [cid for cid, v in r.items() if float(v.get("confidence",1.0)) < 0.5]
        return f"Rebuttals={len(r)} (weak<{0.5}: {len(weak)})."

    def _summarize_judge(self, content: Dict[str, Any]) -> str:
        v = content.get("final_verdict", "No Significant Errors")
        c = content.get("confidence", 0.0)
        return f"Verdict: {v} (conf {c:.2f})"

    def _build_debate_rows(
        self,
        prosecutor_content: Dict[str, Any],
        defense_content: Dict[str, Any],
        reflector_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compact, row-wise table for the UI."""
        case = prosecutor_content.get("case_file", []) or []
        rebuttals = (defense_content or {}).get("rebuttals", {}) or {}
        consensus_points = set((reflector_content or {}).get("consensus_points", []) or [])

        rows = []
        for ch in case:
            cid = ch.get("charge_id")
            r = rebuttals.get(cid, {})
            rows.append({
                "charge_id": cid,
                "error_code": ch.get("error_code"),
                "severity": ch.get("severity"),
                "evidence": ch.get("evidence"),
                "prosecutor_confidence": ch.get("confidence"),
                "defense_rebuttal": r.get("rebuttal_argument"),
                "defense_confidence": r.get("confidence"),
                "reflector_support": (cid in consensus_points)
            })
        return rows

    # ---------- main ----------
    def evaluate_text(self, text: str) -> Dict[str, Any]:
        logger.info("eval_start", extra={"extra": {"trace_id": self.trace_id, "len": len(text)}})

        # turn-by-turn conversation capture
        conversation: List[Dict[str, Any]] = []

        # Prosecutor
        p, p_ms = self._time_call(self.prosecutor.run_on, text, "")
        self._save_json("01_prosecutor.json", {"model": self.models["prosecutor"], "output": p.content})
        logger.debug("prosecutor_done", extra={"extra": {"trace_id": self.trace_id, "ms": round(p_ms,2), "charges": len(p.content.get('case_file', []))}})
        conversation.append({
            "round": 1,
            "role": "Prosecutor",
            "model": self.models["prosecutor"],
            "summary": self._summarize_prosecutor(p.content),
            "content": p.content,
            "duration_ms": p_ms
        })

        # Reflector
        r, r_ms = self._time_call(self.reflector.run_on, text, p.content)
        self._save_json("02_reflector.json", {"model": self.models["reflector"], "output": r.content})
        logger.debug("reflector_done", extra={"extra": {"trace_id": self.trace_id, "ms": round(r_ms,2), "consensus_points": r.content.get('consensus_points', [])}})
        conversation.append({
            "round": 2,
            "role": "Reflector",
            "model": self.models["reflector"],
            "summary": self._summarize_reflector(r.content),
            "content": r.content,
            "duration_ms": r_ms
        })

        # Defense
        d, d_ms = self._time_call(self.defense.run_on, text, p.content, r.content)
        self._save_json("03_defense.json", {"model": self.models["defense"], "output": d.content})
        logger.debug("defense_done", extra={"extra": {"trace_id": self.trace_id, "ms": round(d_ms,2), "rebuttals": list((d.content.get('rebuttals') or {}).keys())}})
        conversation.append({
            "round": 3,
            "role": "Defense",
            "model": self.models["defense"],
            "summary": self._summarize_defense(d.content),
            "content": d.content,
            "duration_ms": d_ms
        })

        # Judge
        j, j_ms = self._time_call(self.judge.run_on, text, p.content, r.content, d.content)
        self._save_json("04_judge.json", {"model": self.models["judge"], "output": j.content})
        logger.debug("judge_done", extra={"extra": {"trace_id": self.trace_id, "ms": round(j_ms,2), "verdict": j.content.get('final_verdict')}})
        conversation.append({
            "round": 4,
            "role": "Judge",
            "model": self.models["judge"],
            "summary": self._summarize_judge(j.content),
            "content": j.content,
            "duration_ms": j_ms
        })

        # Scores & summary
        consensus = calculate_consensus(p.content, d.content, r.content)
        biases = detect_biases(p.content, d.content, text)
        summary = {
            "verdict": j.content.get("final_verdict", "No Significant Errors"),
            "confidence": float(j.content.get("confidence", 0.6)),
            "consensus": float(consensus),
            "has_error": j.content.get("final_verdict") == "Errors Found",
            "num_charges": len(p.content.get("case_file", [])),
            "biases": biases,
        }
        self._save_json("99_summary.json", summary)
        logger.info("eval_complete", extra={"extra": {"trace_id": self.trace_id, **summary}})

        # Debate table rows for UI
        debate_rows = self._build_debate_rows(p.content, d.content, r.content)

        return {
            "summary": summary,
            "round": {
                "prosecutor": asdict(AgentRun("prosecutor", self.models["prosecutor"], p.content, p.confidence, p_ms)),
                "reflector":  asdict(AgentRun("reflector",  self.models["reflector"],  r.content, r.confidence, r_ms)),
                "defense":    asdict(AgentRun("defense",    self.models["defense"],    d.content, d.confidence, d_ms)),
                "judge":      asdict(AgentRun("judge",      self.models["judge"],      j.content, j.confidence, j_ms)),
            },
            "conversation": conversation,   # ← round-by-round log for your round table
            "debate_table": debate_rows     # ← compact rows for a single table render
        }
