# app/graph/round_table_graph.py
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import json
import logging
from dataclasses import dataclass, asdict
from app.agents.base import LLMProvider
from app.config.prompts import PROSECUTOR_PROMPT, DEFENSE_PROMPT, REFLECTOR_PROMPT, JUDGE_PROMPT

logger = logging.getLogger("heda.roundtable")

@dataclass
class ConversationTurn:
    round_num: int
    agent: str
    message: str
    confidence: float
    evidence: List[str]
    charges_filed: List[str] = None
    rebuttals: Dict[str, Any] = None
    consensus_points: List[str] = None
    timestamp: float = None

class RoundTableState(TypedDict):
    text_to_analyze: str
    conversation_history: Annotated[List[ConversationTurn], add_messages]
    current_round: int
    max_rounds: int
    charges: Dict[str, Dict[str, Any]]
    rebuttals: Dict[str, Dict[str, Any]]
    consensus_points: List[str]
    debate_quality: float
    final_verdict: Optional[str]
    final_confidence: float
    should_continue: bool
    participants: List[str]
    speaking_order: List[str]
    current_speaker_idx: int

class RoundTableOrchestrator:
    def __init__(self, provider: LLMProvider, models: Dict[str, str], max_rounds: int = 3):
        self.provider = provider
        self.models = models
        self.max_rounds = max_rounds
        
        # Create the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RoundTableState)
        
        # Add nodes
        workflow.add_node("moderator", self._moderator_node)
        workflow.add_node("prosecutor", self._prosecutor_node)
        workflow.add_node("defense", self._defense_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("round_controller", self._round_controller_node)
        
        # Set entry point
        workflow.set_entry_point("moderator")
        
        # Add edges
        workflow.add_edge("moderator", "prosecutor")
        workflow.add_edge("prosecutor", "round_controller")
        workflow.add_edge("defense", "round_controller") 
        workflow.add_edge("reflector", "round_controller")
        
        # Conditional routing from round_controller
        workflow.add_conditional_edges(
            "round_controller",
            self._should_continue_debate,
            {
                "continue_defense": "defense",
                "continue_reflector": "reflector", 
                "end_debate": "judge"
            }
        )
        
        workflow.add_edge("judge", END)
        
        return workflow.compile()

    def _moderator_node(self, state: RoundTableState) -> RoundTableState:
        """Initialize the round table discussion"""
        logger.info(f"🎯 Moderator: Starting analysis of text ({len(state['text_to_analyze'])} chars)")
        
        state["current_round"] = 1
        state["should_continue"] = True
        state["participants"] = ["prosecutor", "defense", "reflector"]
        state["speaking_order"] = ["prosecutor", "defense", "reflector"]
        state["current_speaker_idx"] = 0
        state["charges"] = {}
        state["rebuttals"] = {}
        state["consensus_points"] = []
        state["conversation_history"] = []
        
        return state

    def _prosecutor_node(self, state: RoundTableState) -> RoundTableState:
        """Prosecutor presents charges or responds to previous discussion"""
        current_round = state["current_round"]
        text = state["text_to_analyze"]
        history = state["conversation_history"]
        
        # Build context from conversation history
        context = self._build_context_for_agent("prosecutor", history)
        
        if current_round == 1:
            prompt = f"""You are the Prosecutor in a reasoning evaluation roundtable. 
Analyze the text and present your initial charges for logical errors.

TEXT TO ANALYZE:
{text}

Return JSON with your charges:
{{"case_file": [{{"charge_id": "c1", "error_code": "E202", "severity": "high", "evidence": "...", "confidence": 0.9}}], "message": "I present these charges..."}}"""
        else:
            prompt = f"""You are the Prosecutor continuing the roundtable discussion.

ORIGINAL TEXT:
{text}

PREVIOUS CONVERSATION:
{context}

Respond to previous arguments and update your position. You may:
- Strengthen existing charges with new evidence
- File additional charges
- Respond to defense rebuttals
- Address reflector concerns

Return JSON: {{"updated_charges": [...], "message": "In response to...", "confidence": 0.85}}"""

        response = self.provider.chat_json(self.models["prosecutor"], "", prompt)
        
        # Update state
        if current_round == 1:
            case_file = response.get("case_file", [])
            for charge in case_file:
                state["charges"][charge["charge_id"]] = charge
        else:
            updated_charges = response.get("updated_charges", [])
            for charge in updated_charges:
                state["charges"][charge["charge_id"]] = charge

        # Add to conversation
        turn = ConversationTurn(
            round_num=current_round,
            agent="prosecutor",
            message=response.get("message", ""),
            confidence=response.get("confidence", 0.7),
            evidence=response.get("evidence", []),
            charges_filed=list(state["charges"].keys())
        )
        state["conversation_history"].append(turn)
        
        logger.info(f"⚖️  Prosecutor Round {current_round}: {len(state['charges'])} total charges")
        return state

    def _defense_node(self, state: RoundTableState) -> RoundTableState:
        """Defense attorney responds to charges and previous discussion"""
        current_round = state["current_round"]
        text = state["text_to_analyze"]
        history = state["conversation_history"]
        charges = state["charges"]
        
        context = self._build_context_for_agent("defense", history)
        
        prompt = f"""You are the Defense Attorney in this roundtable discussion.

ORIGINAL TEXT:
{text}

CURRENT CHARGES:
{json.dumps(charges, indent=2)}

PREVIOUS CONVERSATION:
{context}

Provide your defense. You may:
- Rebut specific charges
- Present alternative interpretations
- Challenge evidence quality
- Respond to prosecutor's latest arguments

Return JSON: {{"rebuttals": {{"c1": {{"rebuttal_argument": "...", "confidence": 0.6}}}}, "message": "I object to...", "overall_confidence": 0.7}}"""

        response = self.provider.chat_json(self.models["defense"], "", prompt)
        
        # Update rebuttals
        new_rebuttals = response.get("rebuttals", {})
        state["rebuttals"].update(new_rebuttals)
        
        turn = ConversationTurn(
            round_num=current_round,
            agent="defense",
            message=response.get("message", ""),
            confidence=response.get("overall_confidence", 0.7),
            evidence=response.get("evidence", []),
            rebuttals=new_rebuttals
        )
        state["conversation_history"].append(turn)
        
        logger.info(f"🛡️  Defense Round {current_round}: {len(state['rebuttals'])} rebuttals")
        return state

    def _reflector_node(self, state: RoundTableState) -> RoundTableState:
        """Reflector analyzes the debate quality and points of consensus"""
        current_round = state["current_round"]
        text = state["text_to_analyze"]
        history = state["conversation_history"]
        charges = state["charges"]
        rebuttals = state["rebuttals"]
        
        context = self._build_context_for_agent("reflector", history)
        
        prompt = f"""You are the Reflector analyzing this roundtable debate.

ORIGINAL TEXT:
{text}

CHARGES VS REBUTTALS:
{json.dumps({"charges": charges, "rebuttals": rebuttals}, indent=2)}

CONVERSATION SO FAR:
{context}

Assess the debate quality and identify consensus points:

Return JSON: {{
    "consensus_points": ["c1", "c3"], 
    "confidence_in_debate_quality": 0.8,
    "message": "I observe that...",
    "areas_needing_clarification": ["charge c2 needs more evidence"],
    "debate_quality_issues": ["prosecutor overconfident on c4"]
}}"""

        response = self.provider.chat_json(self.models["reflector"], "", prompt)
        
        state["consensus_points"] = response.get("consensus_points", [])
        state["debate_quality"] = response.get("confidence_in_debate_quality", 0.7)
        
        turn = ConversationTurn(
            round_num=current_round,
            agent="reflector",
            message=response.get("message", ""),
            confidence=response.get("confidence_in_debate_quality", 0.7),
            evidence=response.get("areas_needing_clarification", []),
            consensus_points=state["consensus_points"]
        )
        state["conversation_history"].append(turn)
        
        logger.info(f"🔍 Reflector Round {current_round}: {len(state['consensus_points'])} consensus points")
        return state

    def _round_controller_node(self, state: RoundTableState) -> RoundTableState:
        """Controls the flow of the roundtable discussion"""
        current_round = state["current_round"]
        max_rounds = state["max_rounds"]
        
        # Check if we should continue the debate
        if current_round >= max_rounds:
            state["should_continue"] = False
            logger.info(f"📝 Round Controller: Max rounds ({max_rounds}) reached, ending debate")
        elif len(state["charges"]) == 0:
            state["should_continue"] = False
            logger.info("📝 Round Controller: No charges filed, ending debate")
        elif state["debate_quality"] < 0.3:
            state["should_continue"] = False
            logger.info("📝 Round Controller: Low debate quality, ending debate")
        else:
            state["current_round"] += 1
            logger.info(f"📝 Round Controller: Continuing to round {state['current_round']}")
        
        return state

    def _should_continue_debate(self, state: RoundTableState) -> str:
        """Determine next step in the debate"""
        if not state["should_continue"]:
            return "end_debate"
        
        current_round = state["current_round"]
        
        # Rotate speakers: prosecutor -> defense -> reflector -> prosecutor...
        if current_round % 3 == 2:  # Defense turn
            return "continue_defense"
        elif current_round % 3 == 0:  # Reflector turn
            return "continue_reflector"
        else:  # Back to prosecutor
            return "continue_prosecutor"

    def _judge_node(self, state: RoundTableState) -> RoundTableState:
        """Judge makes the final decision based on the roundtable discussion"""
        text = state["text_to_analyze"]
        history = state["conversation_history"]
        charges = state["charges"]
        rebuttals = state["rebuttals"]
        consensus_points = state["consensus_points"]
        
        # Build comprehensive summary for judge
        conversation_summary = "\n".join([
            f"Round {turn.round_num} - {turn.agent.upper()}: {turn.message}"
            for turn in history
        ])
        
        prompt = f"""You are the Judge making the final decision in this reasoning evaluation case.

ORIGINAL TEXT:
{text}

FULL ROUNDTABLE CONVERSATION:
{conversation_summary}

FINAL CHARGES:
{json.dumps(charges, indent=2)}

FINAL REBUTTALS:
{json.dumps(rebuttals, indent=2)}

CONSENSUS POINTS: {consensus_points}

Based on the full roundtable discussion, make your final judgment:

Return JSON: {{
    "final_verdict": "Errors Found" | "No Significant Errors",
    "confidence": 0.85,
    "reasoning": "After careful consideration of the roundtable discussion...",
    "sustained_charges": ["c1", "c3"],
    "overruled_charges": ["c2"],
    "key_factors": ["Strong evidence for c1", "Weak rebuttal for c3"]
}}"""

        response = self.provider.chat_json(self.models["judge"], "", prompt)
        
        state["final_verdict"] = response.get("final_verdict", "No Significant Errors")
        state["final_confidence"] = response.get("confidence", 0.6)
        
        turn = ConversationTurn(
            round_num=state["current_round"],
            agent="judge",
            message=response.get("reasoning", ""),
            confidence=state["final_confidence"],
            evidence=response.get("key_factors", [])
        )
        state["conversation_history"].append(turn)
        
        logger.info(f"⚖️  Judge: Final verdict - {state['final_verdict']} (confidence: {state['final_confidence']:.2f})")
        return state

    def _build_context_for_agent(self, agent: str, history: List[ConversationTurn]) -> str:
        """Build conversation context relevant for specific agent"""
        if not history:
            return "No previous conversation."
        
        context_parts = []
        for turn in history[-6:]:  # Last 6 turns for context
            context_parts.append(f"{turn.agent.upper()}: {turn.message}")
        
        return "\n".join(context_parts)

    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """Run the complete roundtable evaluation"""
        initial_state = RoundTableState(
            text_to_analyze=text,
            conversation_history=[],
            current_round=1,
            max_rounds=self.max_rounds,
            charges={},
            rebuttals={},
            consensus_points=[],
            debate_quality=0.7,
            final_verdict=None,
            final_confidence=0.0,
            should_continue=True,
            participants=[],
            speaking_order=[],
            current_speaker_idx=0
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Format results
        return {
            "summary": {
                "verdict": final_state["final_verdict"],
                "confidence": final_state["final_confidence"],
                "has_error": final_state["final_verdict"] == "Errors Found",
                "num_charges": len(final_state["charges"]),
                "num_rebuttals": len(final_state["rebuttals"]),
                "consensus_points": final_state["consensus_points"],
                "debate_quality": final_state["debate_quality"],
                "total_rounds": final_state["current_round"]
            },
            "roundtable_conversation": [asdict(turn) for turn in final_state["conversation_history"]],
            "final_charges": final_state["charges"],
            "final_rebuttals": final_state["rebuttals"],
            "meta": {
                "framework": "HEDA-RoundTable",
                "version": "2.0",
                "graph_type": "langgraph"
            }
        }