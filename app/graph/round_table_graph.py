from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import re
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RoundTableState(TypedDict):
    """State for the round table discussion."""
    original_text: str
    messages: List[Any]
    identified_errors: List[Dict[str, Any]]
    current_round: int
    max_rounds: int
    phase: str
    final_verdict: Optional[Dict[str, Any]]

class RoundTableGraph:
    """Graph-based implementation of roundtable discussion."""
    
    def __init__(self, provider, max_rounds=2):
        self.provider = provider
        self.max_rounds = max_rounds
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the discussion graph."""
        graph = StateGraph(RoundTableState)
        
        # Add nodes
        graph.add_node("moderator", self.moderator_node)
        graph.add_node("prosecutor", self.prosecutor_node)
        graph.add_node("defender", self.defender_node)
        graph.add_node("judge", self.judge_node)
        graph.add_node("jury", self.jury_node)
        
        # Set entry point
        graph.set_entry_point("moderator")
        
        # Add edges
        graph.add_edge("moderator", "prosecutor")
        graph.add_conditional_edges(
            "prosecutor",
            self.should_continue_discussion,
            {
                "defense": "defender",
                "jury": "jury",
                "end": END
            }
        )
        graph.add_edge("defender", "judge")
        graph.add_conditional_edges(
            "judge",
            self.should_start_new_round,
            {
                "prosecutor": "prosecutor",
                "jury": "jury",
                "end": END
            }
        )
        graph.add_edge("jury", END)
        
        return graph.compile()
    
    def moderator_node(self, state: RoundTableState) -> Dict[str, Any]:
        """Moderator introduces topic and manages discussion."""
        logger.info(f"🎯 Moderator: Starting analysis of text ({len(state['original_text'])} chars)")
        
        prompt = f"""As the moderator of this logical analysis discussion, introduce the text we'll be analyzing.
        
Text to analyze:
{state['original_text']}

Provide a brief, neutral introduction of what we're examining today."""
        
        try:
            response = self.provider.generate(prompt, role="moderator")
            
            moderator_message = AIMessage(
                content=response,
                additional_kwargs={"role": "moderator", "round": 0}
            )
            
            return {
                "messages": [moderator_message],
                "current_round": 0,
                "phase": "prosecution"
            }
        except Exception as e:
            logger.error(f"Moderator error: {e}")
            # Return mock response for testing
            return {
                "messages": [AIMessage(
                    content="Let's analyze this text for logical consistency.",
                    additional_kwargs={"role": "moderator", "round": 0}
                )],
                "current_round": 0,
                "phase": "prosecution"
            }
    
    def prosecutor_node(self, state: RoundTableState) -> Dict[str, Any]:
        """Prosecutor identifies logical errors."""
        round_num = state.get("current_round", 0) + 1
        
        prompt = self._build_prosecutor_prompt(state)
        
        try:
            response = self.provider.generate(prompt, role="prosecutor")
            errors = self._parse_errors(response)
            
            prosecutor_message = AIMessage(
                content=response,
                additional_kwargs={
                    "role": "prosecutor", 
                    "round": round_num,
                    "errors_count": len(errors)
                }
            )
            
            logger.info(f"⚖️  Prosecutor Round {round_num}: {len(errors)} total charges")
            
            # Merge new errors with existing ones
            all_errors = state.get("identified_errors", []) + errors
            
            return {
                "messages": [prosecutor_message],
                "identified_errors": all_errors,
                "current_round": round_num,
                "phase": "defense"
            }
        except Exception as e:
            logger.error(f"Prosecutor error: {e}")
            # Return mock response
            mock_errors = [{"type": "logical_fallacy", "severity": "medium", "description": "Potential issue identified"}]
            return {
                "messages": [AIMessage(
                    content="I identify potential logical issues in this text.",
                    additional_kwargs={"role": "prosecutor", "round": round_num}
                )],
                "identified_errors": mock_errors,
                "current_round": round_num,
                "phase": "defense"
            }
    
    def defender_node(self, state: RoundTableState) -> Dict[str, Any]:
        """Defender provides counterarguments."""
        round_num = state.get("current_round", 0)
        
        prompt = self._build_defender_prompt(state)
        
        try:
            response = self.provider.generate(prompt, role="defender")
            
            defender_message = AIMessage(
                content=response,
                additional_kwargs={"role": "defender", "round": round_num}
            )
            
            logger.info(f"🛡️  Defender Round {round_num}: Providing counterarguments")
            
            return {
                "messages": [defender_message],
                "phase": "judge"
            }
        except Exception as e:
            logger.error(f"Defender error: {e}")
            return {
                "messages": [AIMessage(
                    content="I defend the logical consistency of this text.",
                    additional_kwargs={"role": "defender", "round": round_num}
                )],
                "phase": "judge"
            }
    
    def judge_node(self, state: RoundTableState) -> Dict[str, Any]:
        """Judge evaluates arguments from both sides."""
        round_num = state.get("current_round", 0)
        
        prompt = self._build_judge_prompt(state)
        
        try:
            response = self.provider.generate(prompt, role="judge")
            
            judge_message = AIMessage(
                content=response,
                additional_kwargs={"role": "judge", "round": round_num}
            )
            
            logger.info(f"⚖️  Judge Round {round_num}: Evaluating arguments")
            
            return {
                "messages": [judge_message],
                "phase": "decision"
            }
        except Exception as e:
            logger.error(f"Judge error: {e}")
            return {
                "messages": [AIMessage(
                    content="I evaluate the arguments presented.",
                    additional_kwargs={"role": "judge", "round": round_num}
                )],
                "phase": "decision"
            }
    
    def jury_node(self, state: RoundTableState) -> Dict[str, Any]:
        """Jury provides final verdict."""
        prompt = self._build_jury_prompt(state)
        
        try:
            response = self.provider.generate(prompt, role="jury")
            verdict = self._parse_verdict(response)
            
            jury_message = AIMessage(
                content=response,
                additional_kwargs={"role": "jury", "round": state.get("current_round", 0)}
            )
            
            logger.info(f"🎭 Jury: Final verdict reached")
            
            return {
                "messages": [jury_message],
                "final_verdict": verdict,
                "phase": "end"
            }
        except Exception as e:
            logger.error(f"Jury error: {e}")
            return {
                "messages": [AIMessage(
                    content="The jury has reached a verdict.",
                    additional_kwargs={"role": "jury", "round": state.get("current_round", 0)}
                )],
                "final_verdict": {"verdict": "evaluated", "confidence": 0.7},
                "phase": "end"
            }
    
    def should_continue_discussion(self, state: RoundTableState) -> str:
        """Decide whether to continue discussion or move to verdict."""
        errors = state.get("identified_errors", [])
        
        if not errors:
            return "jury"
        
        if state.get("phase") == "defense":
            return "defense"
        
        return "jury"
    
    def should_start_new_round(self, state: RoundTableState) -> str:
        """Decide whether to start a new round or conclude."""
        current_round = state.get("current_round", 0)
        max_rounds = state.get("max_rounds", self.max_rounds)
        
        if current_round >= max_rounds:
            return "jury"
        
        # Check if there are unresolved issues
        errors = state.get("identified_errors", [])
        if errors and any(e.get("severity") == "high" for e in errors):
            return "prosecutor"
        
        return "jury"
    
    def _build_prosecutor_prompt(self, state: RoundTableState) -> str:
        """Build prompt for prosecutor."""
        previous_messages = "\n".join([
            f"{m.additional_kwargs.get('role', 'unknown')}: {m.content[:200]}..."
            for m in state.get("messages", [])[-3:]
        ])
        
        return f"""As the prosecutor, identify logical errors and reasoning issues in this text:

Original text:
{state['original_text']}

Previous discussion:
{previous_messages}

Identify specific logical fallacies, contradictions, or reasoning errors.
Format your response as a list of issues with severity levels (high/medium/low)."""
    
    def _build_defender_prompt(self, state: RoundTableState) -> str:
        """Build prompt for defender."""
        recent_prosecution = None
        for msg in reversed(state.get("messages", [])):
            if msg.additional_kwargs.get("role") == "prosecutor":
                recent_prosecution = msg.content
                break
        
        return f"""As the defender, provide counterarguments to the prosecutor's claims:

Original text:
{state['original_text']}

Prosecutor's claims:
{recent_prosecution}

Defend the logical consistency of the text and address each criticism."""
    
    def _build_judge_prompt(self, state: RoundTableState) -> str:
        """Build prompt for judge."""
        recent_prosecution = None
        recent_defense = None
        
        for msg in reversed(state.get("messages", [])):
            if not recent_defense and msg.additional_kwargs.get("role") == "defender":
                recent_defense = msg.content
            elif not recent_prosecution and msg.additional_kwargs.get("role") == "prosecutor":
                recent_prosecution = msg.content
            if recent_prosecution and recent_defense:
                break
        
        return f"""As the judge, evaluate the arguments from both sides:

Prosecutor's arguments:
{recent_prosecution}

Defender's arguments:
{recent_defense}

Provide a balanced assessment of which arguments are most compelling."""
    
    def _build_jury_prompt(self, state: RoundTableState) -> str:
        """Build prompt for jury."""
        all_messages = "\n".join([
            f"{m.additional_kwargs.get('role', 'unknown')}: {m.content[:200]}..."
            for m in state.get("messages", [])[-5:]
        ])
        
        errors_summary = "\n".join([
            f"- {e.get('type')}: {e.get('description')}"
            for e in state.get("identified_errors", [])[:5]
        ])
        
        return f"""As the jury, provide the final verdict on the logical quality of this text:

Discussion summary:
{all_messages}

Identified issues:
{errors_summary}

Provide a final assessment with:
1. Overall logical quality (score 1-10)
2. Key strengths
3. Key weaknesses
4. Recommendations"""
    
    def _parse_errors(self, response: str) -> List[Dict[str, Any]]:
        """Parse errors from prosecutor response."""
        errors = []
        
        # Simple pattern matching for error extraction
        lines = response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['error', 'fallacy', 'contradiction', 'issue']):
                severity = 'high' if 'serious' in line.lower() or 'major' in line.lower() else 'medium'
                errors.append({
                    'type': 'logical_error',
                    'severity': severity,
                    'description': line.strip()
                })
        
        return errors
    
    def _parse_verdict(self, response: str) -> Dict[str, Any]:
        """Parse verdict from jury response."""
        verdict = {
            'score': 5,
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Try to extract score
        score_match = re.search(r'(\d+)/10|\b([1-9]|10)\b.*score', response.lower())
        if score_match:
            score_str = score_match.group(1) or score_match.group(2)
            try:
                verdict['score'] = int(score_str)
            except:
                pass
        
        return verdict
    
    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """Main evaluation method."""
        initial_state = {
            "original_text": text,
            "messages": [],
            "identified_errors": [],
            "current_round": 0,
            "max_rounds": self.max_rounds,
            "phase": "moderator",
            "final_verdict": None
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "messages": final_state.get("messages", []),
                "identified_errors": final_state.get("identified_errors", []),
                "final_verdict": final_state.get("final_verdict"),
                "mode": "roundtable",
                "rounds_completed": final_state.get("current_round", 0)
            }
        except Exception as e:
            logger.error(f"RoundTable evaluation error: {e}")
            raise