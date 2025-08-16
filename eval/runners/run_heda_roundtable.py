#!/usr/bin/env python3
import json
import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from eval.judges.heda_roundtable_judge import HEDAJudgeSystem, JudgmentTask

async def main():
    if len(sys.argv) < 3:
        print("Usage: python run_heda_roundtable.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Processing {input_file}...")
    judge = HEDAJudgeSystem()
    results = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())
            task = JudgmentTask(
                question=item.get('question', item.get('problem', '')),
                response=item.get('response', item.get('solution', '')),
                ground_truth=item.get('ground_truth', {}),
                difficulty=item.get('difficulty', 'medium')
            )
            
            result = await judge.judge_response(task)
            
            # Format for evaluation pipeline
            formatted_result = {
                'id': item.get('id', f'item_{i}'),
                'response': result,
                'ground_truth': item.get('ground_truth', {}),
                'prediction': result['judgment']
            }
            results.append(formatted_result)
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Processed {len(results)} items, saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
