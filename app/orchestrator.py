# app/orchestrator.py (Updated)
import json, yaml, os, time
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List
from app.agents.base import LLMProvider
from app.graph.round_table_graph import RoundTableOrchestrator
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
    def __init__(self, config_path="app/config/config.yaml", trace_id: Optional[str] = None, use_roundtable: bool = True):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.trace_id = trace_id or os.getenv("TRACE_ID") or "no-trace"
        self.save_artifacts = os.getenv("TRACE_SAVE_ARTIFACTS", "false").lower() == "true"
        self.use_roundtable = use_roundtable
        
        self.provider = LLMProvider(use_openrouter=bool(self.config.get("use_openrouter", False)))
        
        roles_cfg = self.config.get("roles") or {}
        tiers = self.config["model_tiers"]
        self.models = {
            "prosecutor": roles_cfg.get("prosecutor", tiers["balanced"][0]),
            "reflector":  roles_cfg.get("reflector",  tiers["balanced"][0]),
            "defense":    roles_cfg.get("defense",    tiers["balanced"][0]),
            "judge":      roles_cfg.get("judge",      tiers["premium"][0]),
        }
        
        # Initialize round table orchestrator if enabled
        if self.use_roundtable:
            max_rounds = self.config.get("max_rounds", 3)
            self.round_table = RoundTableOrchestrator(
                provider=self.provider,
                models=self.models,
                max_rounds=max_rounds
            )
        else:
            # Keep original agents for fallback
            from app.agents.prosecutor import ProsecutorAgent
            from app.agents.reflector import ReflectorAgent
            from app.agents.defense import DefenseAgent
            from app.agents.judge import JudgeAgent
            
            self.prosecutor = ProsecutorAgent("prosecutor", self.models["prosecutor"], "", self.provider)
            self.reflector  = ReflectorAgent("reflector",  self.models["reflector"],  "", self.provider)
            self.defense    = DefenseAgent("defense",    self.models["defense"],    "", self.provider)
            self.judge      = JudgeAgent("judge",      self.models["judge"],      "", self.provider)
        
        logger.debug("orchestrator_init", extra={"extra": {
            "trace_id": self.trace_id, 
            "models": self.models,
            "roundtable_mode": self.use_roundtable
        }})

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

    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """Main evaluation method that can use either roundtable or traditional approach"""
        logger.info("eval_start", extra={"extra": {
            "trace_id": self.trace_id, 
            "len": len(text),
            "mode": "roundtable" if self.use_roundtable else "traditional"
        }})

        if self.use_roundtable:
            return self._evaluate_with_roundtable(text)
        else:
            return self._evaluate_traditional(text)

    def _evaluate_with_roundtable(self, text: str) -> Dict[str, Any]:
        """Use the new LangGraph roundtable approach"""
        start_time = time.perf_counter()
        
        try:
            # Run the round table discussion
            result = self.round_table.evaluate_text(text)
            
            # Save artifacts
            self._save_json("roundtable_full_result.json", result)
            
            # Extract conversation for backward compatibility
            conversation = result.get("roundtable_conversation", [])
            
            # Build summary compatible with existing UI
            summary = result["summary"]
            summary.update({
                "biases": detect_biases(
                    {"case_file": list(result.get("final_charges", {}).values())}, 
                    {"rebuttals": result.get("final_rebuttals", {})}, 
                    text
                ),
                "consensus": len(summary.get("consensus_points", [])) / max(summary.get("num_charges", 1), 1)
            })
            
            # Format for UI compatibility
            total_time = (time.perf_counter() - start_time) * 1000.0
            
            # Build round data for backward compatibility
            round_data = {
                "prosecutor": AgentRun("prosecutor", self.models["prosecutor"], 
                                     {"case_file": list(result.get("final_charges", {}).values())}, 
                                     0.8, total_time * 0.3),
                "reflector": AgentRun("reflector", self.models["reflector"],
                                    {"consensus_points": summary.get("consensus_points", []),
                                     "confidence_in_debate_quality": summary.get("debate_quality", 0.7)},
                                    summary.get("debate_quality", 0.7), total_time * 0.2),
                "defense": AgentRun("defense", self.models["defense"],
                                  {"rebuttals": result.get("final_rebuttals", {})},
                                  0.7, total_time * 0.3),
                "judge": AgentRun("judge", self.models["judge"],
                                {"final_verdict": summary["verdict"],
                                 "confidence": summary["confidence"],
                                 "reasoning": f"After {summary.get('total_rounds', 1)} rounds of discussion"},
                                summary["confidence"], total_time * 0.2)
            }

            # Build conversation table for UI
            conversation_table = []
            for i, turn in enumerate(conversation, 1):
                conversation_table.append({
                    "round": turn.get("round_num", i),
                    "role": turn.get("agent", "unknown").title(),
                    "model": self.models.get(turn.get("agent", "unknown"), "unknown"),
                    "summary": turn.get("message", "")[:100] + "..." if len(turn.get("message", "")) > 100 else turn.get("message", ""),
                    "content": {
                        "message": turn.get("message", ""),
                        "confidence": turn.get("confidence", 0.0),
                        "evidence": turn.get("evidence", [])
                    },
                    "duration_ms": total_time / len(conversation) if conversation else 0
                })

            # Build debate table rows
            debate_rows = self._build_debate_rows_from_roundtable(result)

            logger.info("roundtable_eval_complete", extra={"extra": {
                "trace_id": self.trace_id, 
                "verdict": summary["verdict"],
                "confidence": summary["confidence"],
                "total_rounds": summary.get("total_rounds", 1),
                "conversation_turns": len(conversation)
            }})

            return {
                "summary": summary,
                "round": {k: asdict(v) for k, v in round_data.items()},
                "conversation": conversation_table,
                "debate_table": debate_rows,
                "roundtable_conversation": conversation,  # Full roundtable data
                "meta": result.get("meta", {}),
                "performance": {
                    "total_duration_ms": total_time,
                    "mode": "roundtable",
                    "langgraph_version": True
                }
            }
            
        except Exception as e:
            logger.error("roundtable_error", extra={"extra": {
                "trace_id": self.trace_id, 
                "error": str(e)
            }})
            # Fallback to traditional approach
            logger.info("falling_back_to_traditional")
            return self._evaluate_traditional(text)

    def _evaluate_traditional(self, text: str) -> Dict[str, Any]:
        """Original sequential approach for comparison/fallback"""
        conversation = []

        # Prosecutor
        p, p_ms = self._time_call(self.prosecutor.run_on, text, "")
        self._save_json("01_prosecutor.json", {"model": self.models["prosecutor"], "output": p.content})
        conversation.append({
            "round": 1, "role": "Prosecutor", "model": self.models["prosecutor"],
            "summary": self._summarize_prosecutor(p.content),
            "content": p.content, "duration_ms": p_ms
        })

        # Reflector
        r, r_ms = self._time_call(self.reflector.run_on, text, p.content)
        self._save_json("02_reflector.json", {"model": self.models["reflector"], "output": r.content})
        conversation.append({
            "round": 2, "role": "Reflector", "model": self.models["reflector"],
            "summary": self._summarize_reflector(r.content),
            "content": r.content, "duration_ms": r_ms
        })

        # Defense
        d, d_ms = self._time_call(self.defense.run_on, text, p.content, r.content)
        self._save_json("03_defense.json", {"model": self.models["defense"], "output": d.content})
        conversation.append({
            "round": 3, "role": "Defense", "model": self.models["defense"],
            "summary": self._summarize_defense(d.content),
            "content": d.content, "duration_ms": d_ms
        })

        # Judge
        j, j_ms = self._time_call(self.judge.run_on, text, p.content, r.content, d.content)
        self._save_json("04_judge.json", {"model": self.models["judge"], "output": j.content})
        conversation.append({
            "round": 4, "role": "Judge", "model": self.models["judge"],
            "summary": self._summarize_judge(j.content),
            "content": j.content, "duration_ms": j_ms
        })

        # Summary
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

        debate_rows = self._build_debate_rows(p.content, d.content, r.content)

        return {
            "summary": summary,
            "round": {
                "prosecutor": asdict(AgentRun("prosecutor", self.models["prosecutor"], p.content, p.confidence, p_ms)),
                "reflector":  asdict(AgentRun("reflector",  self.models["reflector"],  r.content, r.confidence, r_ms)),
                "defense":    asdict(AgentRun("defense",    self.models["defense"],    d.content, d.confidence, d_ms)),
                "judge":      asdict(AgentRun("judge",      self.models["judge"],      j.content, j.confidence, j_ms)),
            },
            "conversation": conversation,
            "debate_table": debate_rows,
            "meta": {"framework": "HEDA-Traditional", "version": "1.0"}
        }

    def _build_debate_rows_from_roundtable(self, roundtable_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build debate table compatible with UI from roundtable results"""
        charges = roundtable_result.get("final_charges", {})
        rebuttals = roundtable_result.get("final_rebuttals", {})
        consensus_points = set(roundtable_result.get("summary", {}).get("consensus_points", []))
        
        rows = []
        for charge_id, charge in charges.items():
            rebuttal = rebuttals.get(charge_id, {})
            rows.append({
                "charge_id": charge_id,
                "error_code": charge.get("error_code"),
                "severity": charge.get("severity"),
                "evidence": charge.get("evidence"),
                "prosecutor_confidence": charge.get("confidence"),
                "defense_rebuttal": rebuttal.get("rebuttal_argument"),
                "defense_confidence": rebuttal.get("confidence"),
                "reflector_support": charge_id in consensus_points
            })
        return rows

    def _build_debate_rows(self, prosecutor_content: Dict[str, Any], defense_content: Dict[str, Any], reflector_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Original debate table builder for traditional mode"""
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

    # Keep existing summarizer methods for backward compatibility
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