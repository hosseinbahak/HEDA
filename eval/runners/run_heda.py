import argparse
import json
import os
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.orchestrator import Orchestrator
from app.provider import Provider
from app.utils.logger import get_logger
from app.utils.conversation_display import ConversationDisplay

logger = get_logger(__name__)

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def serialize_message(msg: Any) -> Dict[str, Any]:
    """Convert a message object to a JSON-serializable dictionary."""
    if hasattr(msg, 'content') and hasattr(msg, 'additional_kwargs'):
        return {
            'content': msg.content,
            'role': msg.additional_kwargs.get('role', 'unknown'),
            'round': msg.additional_kwargs.get('round', 0),
            'type': 'message'
        }
    elif hasattr(msg, '__dict__'):
        return msg.__dict__
    else:
        return str(msg)

def make_json_serializable(obj: Any) -> Any:
    """Recursively make an object JSON serializable."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'content') and hasattr(obj, 'additional_kwargs'):
        # Handle LangChain messages
        return serialize_message(obj)
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

def run_with_orchestrator(
    text: str,
    orch: Orchestrator,
    display_conversation: bool = True,
    console_display: ConversationDisplay = None
) -> Dict[str, Any]:
    """Run evaluation with orchestrator and optional conversation display."""
    try:
        result = orch.evaluate_text(text)
        
        # Display conversation if in roundtable mode and display is enabled
        if orch.mode == "roundtable" and display_conversation and console_display:
            console_display.display_complete_analysis(result)
        
        # Make result JSON serializable
        return make_json_serializable(result)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "mode": orch.mode,
            "status": "failed"
        }

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            # Ensure the result is JSON serializable
            serializable_result = make_json_serializable(result)
            f.write(json.dumps(serializable_result) + '\n')
    
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run HEDA evaluation")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--mode", default="traditional", 
                       choices=["traditional", "roundtable"],
                       help="Evaluation mode")
    parser.add_argument("--max_rounds", type=int, default=2,
                       help="Maximum discussion rounds for roundtable mode")
    parser.add_argument("--display_conversation", action="store_true", default=True,
                       help="Display conversation in console")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--api_key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Setup API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("No API key provided. Using mock provider.")
    
    # Initialize provider and orchestrator
    provider = Provider(api_key=api_key)
    orchestrator = Orchestrator(
        provider=provider,
        mode=args.mode,
        max_rounds=args.max_rounds,
        verbose=args.verbose
    )
    
    # Initialize display if needed
    console_display = ConversationDisplay() if args.display_conversation else None
    
    # Load dataset
    samples = load_dataset(args.input)
    logger.info(f"Loaded {len(samples)} samples from {args.input}")
    
    # Process samples
    results = []
    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{i}")
        text = sample.get("text", "")
        
        logger.info(f"Processing sample ID: {sample_id}, mode: {args.mode}")
        
        if console_display:
            console_display.console.print(f"\n[bold cyan]Processing: {sample_id}[/bold cyan]")
            console_display.console.print(f"[dim]Text: {text[:100]}...[/dim]\n")
        
        result = run_with_orchestrator(
            text=text,
            orch=orchestrator,
            display_conversation=args.display_conversation,
            console_display=console_display
        )
        
        # Add metadata to result
        result["sample_id"] = sample_id
        result["original_text"] = text
        results.append(result)
    
    # Save results
    save_results(results, args.out)
    
    logger.info(f"Evaluation complete. Processed {len(results)} samples.")
    
    if console_display:
        console_display.console.print(
            f"\n[bold green]✅ Evaluation complete![/bold green]"
            f"\nProcessed: {len(results)} samples"
            f"\nOutput: {args.out}"
        )

if __name__ == "__main__":
    main()