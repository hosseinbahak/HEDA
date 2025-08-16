#!/usr/bin/env python3
# eval/runners/run_single_llm.py
import argparse, json, os, time
from app.agents.base import LLMProvider

def evaluate_with_single_llm(provider, model, text):
    """Evaluate using single LLM as judge"""
    prompt = f"""You are an expert reasoning evaluator. Analyze the following text for logical errors, factual inaccuracies, or flawed reasoning.

TEXT TO ANALYZE:
{text}

Provide your evaluation in JSON format:
{{
    "has_error": true/false,
    "confidence": 0.0-1.0,
    "verdict": "Errors Found" or "No Significant Errors", 
    "reasoning": "Explanation of your decision",
    "error_types": ["logical", "factual", "calculation"] // if has_error is true
}}"""

    try:
        response = provider.chat_json(model, "", prompt)
        return {
            "has_error": response.get("has_error", False),
            "verdict": response.get("verdict", "No Significant Errors"),
            "confidence": float(response.get("confidence", 0.5)),
            "reasoning": response.get("reasoning", ""),
            "error_types": response.get("error_types", [])
        }
    except Exception as e:
        return {
            "has_error": False,
            "verdict": "Error in evaluation",
            "confidence": 0.0,
            "reasoning": f"System error: {e}",
            "error_types": []
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True) 
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    
    provider = LLMProvider(use_openrouter=True)
    
    with open(args.input, 'r') as fin, open(args.out, 'w') as fout:
        for line in fin:
            record = json.loads(line.strip())
            text = record.get("prompt", "")
            
            result = evaluate_with_single_llm(provider, args.model, text)
            
            output = {
                "id": record["id"],
                "prediction": result,
                "meta": {"framework": f"Single-{args.model}", "version": "1.0"}
            }
            fout.write(json.dumps(output) + "\n")
            print(f"Processed: {record['id']}")

if __name__ == "__main__":
    main()
