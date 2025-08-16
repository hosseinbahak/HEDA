#!/usr/bin/env python3
# eval/run_comprehensive_evaluation.py
"""
Comprehensive evaluation script that runs multiple systems and generates comparison report.
Supports HEDA (traditional), HEDA-RoundTable, and baseline single-LLM judges.
"""

import argparse
import json
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import concurrent.futures
from dataclasses import dataclass

@dataclass
class SystemConfig:
    name: str
    runner_script: str
    extra_args: Dict[str, str]
    description: str

class EvaluationRunner:
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define systems to evaluate
        self.systems = {
            "HEDA-Traditional": SystemConfig(
                name="HEDA-Traditional",
                runner_script="eval/runners/run_heda.py",
                extra_args={"mode": "traditional"},
                description="Original sequential HEDA with Prosecutor→Reflector→Defense→Judge"
            ),
            "HEDA-RoundTable": SystemConfig(
                name="HEDA-RoundTable", 
                runner_script="eval/runners/run_heda.py",
                extra_args={"mode": "roundtable"},
                description="New LangGraph-based roundtable discussion approach"
            ),
            "GPT4-Single": SystemConfig(
                name="GPT4-Single",
                runner_script="eval/runners/run_single_llm.py",
                extra_args={"model": "openai/gpt-4o"},
                description="Single GPT-4 as judge baseline"
            ),
            "Claude-Single": SystemConfig(
                name="Claude-Single",
                runner_script="eval/runners/run_single_llm.py", 
                extra_args={"model": "anthropic/claude-3-5-sonnet-20240620"},
                description="Single Claude-3.5-Sonnet as judge baseline"
            ),
            "Gemini-Single": SystemConfig(
                name="Gemini-Single",
                runner_script="eval/runners/run_single_llm.py",
                extra_args={"model": "google/gemini-1.5-pro"},
                description="Single Gemini-1.5-Pro as judge baseline"
            )
        }
    
    def run_system_evaluation(self, system_name: str, timeout_per_sample: int = 120) -> Dict[str, Any]:
        """Run evaluation for a single system"""
        print(f"\n🚀 Running evaluation for {system_name}...")
        
        config = self.systems[system_name]
        output_file = self.output_dir / f"{system_name.lower()}_results.jsonl"
        
        # Build command
        cmd = [
            "python", config.runner_script,
            "--input", self.dataset_path,
            "--out", str(output_file)
        ]
        
        # Add extra arguments
        for key, value in config.extra_args.items():
            cmd.extend([f"--{key}", value])
        
        # Set environment variables
        env = os.environ.copy()
        env["RUNNER_SAMPLE_TIMEOUT"] = str(timeout_per_sample)
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600,  # 1 hour max per system
                env=env
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Count successful predictions
                sample_count = 0
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        sample_count = sum(1 for _ in f)
                
                print(f"✅ {system_name} completed: {sample_count} samples in {duration:.1f}s")
                return {
                    "status": "success",
                    "duration": duration,
                    "sample_count": sample_count,
                    "output_file": str(output_file),
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"❌ {system_name} failed with code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return {
                    "status": "failed",
                    "duration": duration,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        
        except subprocess.TimeoutExpired:
            print(f"⏰ {system_name} timed out after 1 hour")
            return {"status": "timeout", "duration": 3600}
        
        except Exception as e:
            print(f"💥 {system_name} crashed: {e}")
            return {"status": "error", "error": str(e), "duration": time.time() - start_time}
    
    def run_all_systems(self, parallel: bool = False, selected_systems: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run evaluation for all systems"""
        systems_to_run = selected_systems or list(self.systems.keys())
        results = {}
        
        print(f"🎯 Running evaluation on {len(systems_to_run)} systems")
        print(f"📊 Dataset: {self.dataset_path}")
        print(f"📁 Output: {self.output_dir}")
        
        if parallel and len(systems_to_run) > 1:
            print("🏃 Running systems in parallel...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_system = {
                    executor.submit(self.run_system_evaluation, system): system 
                    for system in systems_to_run
                }
                
                for future in concurrent.futures.as_completed(future_to_system):
                    system = future_to_system[future]
                    try:
                        results[system] = future.result()
                    except Exception as e:
                        print(f"💥 {system} crashed during parallel execution: {e}")
                        results[system] = {"status": "error", "error": str(e)}
        else:
            print("🚶 Running systems sequentially...")
            for system in systems_to_run:
                results[system] = self.run_system_evaluation(system)
        
        return results
    
    def compute_metrics(self, force_recompute: bool = False) -> str:
        """Compute comprehensive metrics for all completed systems"""
        print("\n📊 Computing comprehensive metrics...")
        
        # Find completed result files
        result_files = []
        for system_name in self.systems.keys():
            result_file = self.output_dir / f"{system_name.lower()}_results.jsonl"
            if result_file.exists():
                result_files.append(f"{system_name}:{result_file}")
        
        if not result_files:
            print("❌ No result files found for metrics computation")
            return ""
        
        # Run comprehensive metrics
        metrics_script = "eval/metrics/comprehensive_metrics.py"
        report_file = self.output_dir / "comprehensive_report.html"
        json_file = self.output_dir / "metrics.json"
        
        cmd = [
            "python", metrics_script,
            "--gold", self.dataset_path,
            "--systems", *result_files,
            "--report", str(report_file),
            "--json-output", str(json_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Metrics computed successfully")
                print(f"📄 HTML Report: {report_file}")
                print(f"📄 JSON Metrics: {json_file}")
                print("\n" + result.stdout)
                return str(report_file)
            else:
                print(f"❌ Metrics computation failed: {result.stderr}")
                return ""
        except Exception as e:
            print(f"💥 Metrics computation crashed: {e}")
            return ""
    
    def create_baseline_runners(self):
        """Create runner scripts for baseline single-LLM systems"""
        single_llm_runner = self.output_dir.parent / "eval" / "runners" / "run_single_llm.py"
        single_llm_runner.parent.mkdir(parents=True, exist_ok=True)
        
        single_llm_code = '''#!/usr/bin/env python3
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
            fout.write(json.dumps(output) + "\\n")
            print(f"Processed: {record['id']}")

if __name__ == "__main__":
    main()
'''
        
        with open(single_llm_runner, 'w') as f:
            f.write(single_llm_code)
        
        os.chmod(single_llm_runner, 0o755)
        print(f"📝 Created baseline runner: {single_llm_runner}")
        
        # Update HEDA runner to support modes
        heda_runner = single_llm_runner.parent / "run_heda.py"
        if heda_runner.exists():
            # Patch the existing runner to support roundtable mode
            with open(heda_runner, 'r') as f:
                content = f.read()
            
            # Add mode support
            if '--mode' not in content:
                content = content.replace(
                    'parser.add_argument("--out"',
                    'parser.add_argument("--mode", choices=["traditional", "roundtable"], default="roundtable", help="Evaluation mode")\n    parser.add_argument("--out"'
                )
                content = content.replace(
                    'orch = Orchestrator()',
                    'orch = Orchestrator(use_roundtable=(args.mode == "roundtable"))'
                )
                
                with open(heda_runner, 'w') as f:
                    f.write(content)
                print(f"📝 Updated HEDA runner with mode support")

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive LLM judge evaluation")
    parser.add_argument("--dataset", required=True, help="Enhanced reasoning dataset JSONL")
    parser.add_argument("--output-dir", default="results/comprehensive_eval", help="Output directory")
    parser.add_argument("--systems", nargs="*", help="Specific systems to run (default: all)")
    parser.add_argument("--parallel", action="store_true", help="Run systems in parallel")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per sample (seconds)")
    parser.add_argument("--metrics-only", action="store_true", help="Only compute metrics, skip evaluation")
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(args.dataset, args.output_dir)
    
    if not args.metrics_only:
        # Create baseline runners if needed
        runner.create_baseline_runners()
        
        # Run evaluations
        results = runner.run_all_systems(
            parallel=args.parallel,
            selected_systems=args.systems
        )
        
        # Save run summary
        summary_file = runner.output_dir / "run_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "dataset": args.dataset,
                "systems": {k: v for k, v in results.items()},
                "config": {
                    "parallel": args.parallel,
                    "timeout": args.timeout
                }
            }, f, indent=2)
        
        print(f"\n📊 Run Summary:")
        for system, result in results.items():
            status = result["status"]
            duration = result.get("duration", 0)
            print(f"  {system}: {status} ({duration:.1f}s)")
    
    # Compute comprehensive metrics
    report_path = runner.compute_metrics()
    
    if report_path:
        print(f"\n🎉 Evaluation complete!")
        print(f"📄 Open the report: {report_path}")
        print(f"📁 All results in: {args.output_dir}")
    else:
        print("❌ Evaluation completed but metrics computation failed")

if __name__ == "__main__":
    main()