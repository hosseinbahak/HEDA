import asyncio
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from eval.judges.heda_roundtable_judge import HEDABatchJudge, JudgmentTask

async def main():
    # Load evaluation dataset
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/datasets/enhanced_reasoning.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/heda_judgments.jsonl"
    
    tasks = []
    with open(dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            task = JudgmentTask(
                question=item.get('question', ''),
                response=item.get('response', ''),
                ground_truth=item.get('ground_truth', {}),
                error_type=item.get('error_type', 'unknown'),
                difficulty=item.get('difficulty', 'medium'),
                domain=item.get('domain', 'general')
            )
            tasks.append(task)
    
    # Configure HEDA judge
    config = {
        'model': 'gpt-4',
        'max_rounds': 3,
        'consensus_threshold': 0.7,
        'jury_size': 3,
        'enable_attorney': True,
        'track_tokens': True
    }
    
    # Run batch judgment
    judge = HEDABatchJudge(config)
    results = await judge.judge_batch(tasks)
    
    # Save results
    judge.save_results(output_path)
    
    # Print summary
    stats = judge.get_summary_statistics()
    print("\n📊 Judgment Summary:")
    print(f"  Total Judged: {stats['total_judged']}")
    print(f"  Errors Found: {stats['errors_found']} ({stats['error_rate']:.1%})")
    print(f"  Avg Confidence: {stats['average_confidence']:.2f}")
    print(f"  Consensus Rate: {stats['consensus_reached_rate']:.1%}")
    print(f"  Avg Rounds: {stats['average_discussion_rounds']:.1f}")

if __name__ == "__main__":
    asyncio.run(main())