import argparse
import json
import logging
import os
from typing import Dict, Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("heda")

# Mock LLM provider (replace with actual OpenRouter API calls if needed)
class MockLLMProvider:
    def chat_json(self, model: str, system: str, user: str) -> Dict[str, Any]:
        try:
            # Mock response for demonstration
            return {
                "has_error": True,
                "confidence": 0.9,
                "explanation": "Mock response: detected logical error"
            }
        except Exception as e:
            logger.error(f"LLM provider error: {e}")
            return {"has_error": True, "confidence": 0.5, "explanation": f"Error in LLM: {e}"}

# Base Agent class
class BaseAgent:
    def __init__(self, llm_provider: MockLLMProvider):
        self.llm_provider = llm_provider

    def evaluate(self, text: str, context: str = "") -> Dict[str, Any]:
        try:
            system_prompt = "Evaluate the reasoning in the provided text for errors."
            user_prompt = f"Text: {text}\nContext: {context}"
            return self.llm_provider.chat_json("mock-model", system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Agent evaluation error: {e}")
            return {"has_error": True, "confidence": 0.5, "explanation": f"Evaluation failed: {e}"}

# RoundTable evaluation logic
def run_roundtable_evaluation(sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    try:
        llm_provider = MockLLMProvider()
        agent = BaseAgent(llm_provider)
        
        # Validate sample
        if not isinstance(sample, dict) or "id" not in sample or "prompt" not in sample:
            logger.error(f"Invalid sample format: {sample}")
            return {
                "id": sample.get("id", "unknown"),
                "prediction": {
                    "has_error": True,
                    "confidence": 0.5,
                    "explanation": "Invalid sample format"
                }
            }
        
        # Define state as a dictionary
        state = {
            "text": sample["prompt"],
            "id": sample["id"],
            "has_error": None,
            "confidence": 0.5,
            "explanation": "",
            "rounds": 0,
            "max_rounds": args.max_rounds,
            "consensus": False
        }
        
        logger.debug(f"Initial state: {state}")
        
        # Define LangGraph workflow
        graph = StateGraph(dict)
        
        def agent1_node(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = agent.evaluate(state["text"])
                logger.debug(f"Agent 1 result: {result}")
                return {
                    "text": state["text"],  # Preserve text
                    "id": state["id"],      # Preserve id
                    "has_error": result.get("has_error", True),
                    "confidence": result.get("confidence", 0.9),
                    "explanation": result.get("explanation", "Agent 1 evaluation completed"),
                    "rounds": state["rounds"] + 1,
                    "max_rounds": state["max_rounds"],  # Preserve max_rounds
                    "consensus": state["consensus"]     # Preserve consensus
                }
            except Exception as e:
                logger.error(f"Agent 1 node error: {e}")
                return {
                    "text": state["text"],
                    "id": state["id"],
                    "has_error": True,
                    "confidence": 0.5,
                    "explanation": f"Agent 1 failed: {e}",
                    "rounds": state["rounds"] + 1,
                    "max_rounds": state["max_rounds"],
                    "consensus": state["consensus"]
                }
        
        def agent2_node(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if "text" not in state:
                    logger.error("Missing 'text' key in state")
                    return {
                        "text": state.get("text", ""),
                        "id": state.get("id", "unknown"),
                        "has_error": True,
                        "confidence": 0.5,
                        "explanation": "Agent 2 failed: Missing text key",
                        "rounds": state.get("rounds", 0) + 1,
                        "max_rounds": state.get("max_rounds", 10),
                        "consensus": False
                    }
                result = agent.evaluate(state["text"], context=state["explanation"])
                logger.debug(f"Agent 2 result: {result}")
                return {
                    "text": state["text"],
                    "id": state["id"],
                    "has_error": result.get("has_error", True),
                    "confidence": result.get("confidence", 0.9),
                    "explanation": result.get("explanation", "Agent 2 evaluation completed"),
                    "rounds": state["rounds"] + 1,
                    "max_rounds": state["max_rounds"],
                    "consensus": state["has_error"] == result.get("has_error", True)
                }
            except Exception as e:
                logger.error(f"Agent 2 node error: {e}")
                return {
                    "text": state.get("text", ""),
                    "id": state.get("id", "unknown"),
                    "has_error": True,
                    "confidence": 0.5,
                    "explanation": f"Agent 2 failed: {e}",
                    "rounds": state.get("rounds", 0) + 1,
                    "max_rounds": state.get("max_rounds", 10),
                    "consensus": False
                }
        
        # Add nodes and edges
        graph.add_node("agent1", agent1_node)
        graph.add_node("agent2", agent2_node)
        graph.set_entry_point("agent1")
        graph.add_edge("agent1", "agent2")
        graph.add_edge("agent2", END)
        
        # Compile and run the graph
        app = graph.compile()
        final_state = app.invoke(state)
        logger.debug(f"Final state: {final_state}")
        
        return {
            "id": sample["id"],
            "prediction": {
                "has_error": final_state["has_error"],
                "confidence": final_state["confidence"],
                "explanation": final_state["explanation"]
            }
        }
    except Exception as e:
        logger.error(f"Roundtable evaluation error for sample {sample.get('id', 'unknown')}: {e}")
        return {
            "id": sample.get("id", "unknown"),
            "prediction": {
                "has_error": True,
                "confidence": 0.5,
                "explanation": f"Roundtable evaluation failed: {e}"
            }
        }

# Traditional evaluation logic
def run_traditional_evaluation(sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    try:
        llm_provider = MockLLMProvider()
        agent = BaseAgent(llm_provider)
        result = agent.evaluate(sample["prompt"])
        return {
            "id": sample["id"],
            "prediction": {
                "has_error": result.get("has_error", True),
                "confidence": result.get("confidence", 0.9),
                "explanation": result.get("explanation", "Traditional evaluation completed")
            }
        }
    except Exception as e:
        logger.error(f"Traditional evaluation error for sample {sample.get('id', 'unknown')}: {e}")
        return {
            "id": sample.get("id", "unknown"),
            "prediction": {
                "has_error": True,
                "confidence": 0.5,
                "explanation": f"Traditional evaluation failed: {e}"
            }
        }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run HEDA evaluation")
    parser.add_argument("--input", required=True, help="Input dataset file (JSONL)")
    parser.add_argument("--out", required=True, help="Output results file (JSONL)")
    parser.add_argument("--mode", choices=["roundtable", "traditional"], default="roundtable", 
                        help="Evaluation mode: roundtable or traditional")
    parser.add_argument("--max_rounds", type=int, default=10, help="Max rounds for roundtable")
    parser.add_argument("--consensus_threshold", type=float, default=0.7, help="Consensus threshold")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file {args.input} does not exist")
        exit(1)

    # Create output directory if it doesn't exist
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Process dataset
    results = []
    with open(args.input, "r") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                logger.info(f"Processing sample ID: {sample['id']}, mode: {args.mode}")
                
                # Run evaluation based on mode
                if args.mode == "roundtable":
                    result = run_roundtable_evaluation(sample, args)
                else:
                    result = run_traditional_evaluation(sample, args)
                results.append(result)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in input file: {line.strip()}, error: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue

    # Save results
    with open(args.out, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Evaluation completed, output saved to {args.out}, processed {len(results)} samples")

if __name__ == "__main__":
    main()