# eval/metrics/comprehensive_metrics.py
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LLMJudgeMetrics:
    def __init__(self, gold_path: str):
        """Initialize with gold standard data"""
        self.gold = self._load_gold(gold_path)
        self.results = {}
        
    def _load_gold(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Load gold standard with rich annotations"""
        gold = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                gold[item['id']] = {
                    'has_error': bool(item.get('gold_has_error', False)),
                    'error_type': item.get('error_type', 'unknown'),
                    'difficulty': item.get('difficulty', 'medium'),
                    'domain': item.get('domain', 'general'),
                    'gold_confidence': item.get('gold_confidence', 1.0),
                    'explanation': item.get('explanation', '')
                }
        return gold
    
    def add_system_results(self, system_name: str, predictions_path: str):
        """Add results from a system"""
        predictions = []
        with open(predictions_path, 'r', encoding='utf-8') as f:
            for line in f:
                predictions.append(json.loads(line.strip()))
        
        self.results[system_name] = predictions
        print(f"Added {len(predictions)} predictions for {system_name}")
    
    def compute_basic_metrics(self, system_name: str) -> Dict[str, float]:
        """Compute standard classification metrics"""
        predictions = self.results[system_name]
        
        y_true, y_pred, y_prob, sample_ids = [], [], [], []
        
        for pred in predictions:
            sample_id = pred['id']
            if sample_id not in self.gold:
                continue
                
            y_true.append(self.gold[sample_id]['has_error'])
            y_pred.append(pred['prediction']['has_error'])
            y_prob.append(pred['prediction']['confidence'])
            sample_ids.append(sample_id)
        
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=True
        )
        
        # Confidence calibration (Expected Calibration Error)
        ece = self._expected_calibration_error(y_true, y_prob)
        
        # Area under ROC curve
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5  # Default for constant predictions
        
        # Confidence-aware metrics
        confident_correct = np.mean((y_true == y_pred) & (y_prob > 0.8))
        uncertain_correct = np.mean((y_true == y_pred) & (y_prob <= 0.6))
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'ece': float(ece),
            'confident_correct': float(confident_correct),
            'uncertain_correct': float(uncertain_correct),
            'sample_count': len(y_true)
        }
    
    def compute_advanced_metrics(self, system_name: str) -> Dict[str, Any]:
        """Compute advanced metrics specific to reasoning evaluation"""
        predictions = self.results[system_name]
        
        # Group by difficulty, domain, error type
        metrics_by_difficulty = defaultdict(list)
        metrics_by_domain = defaultdict(list)
        metrics_by_error_type = defaultdict(list)
        
        reasoning_depth_scores = []
        explanation_quality_scores = []
        
        for pred in predictions:
            sample_id = pred['id']
            if sample_id not in self.gold:
                continue
                
            gold_item = self.gold[sample_id]
            pred_item = pred['prediction']
            
            is_correct = gold_item['has_error'] == pred_item['has_error']
            confidence = pred_item['confidence']
            
            # Group metrics
            difficulty = gold_item['difficulty']
            domain = gold_item['domain'] 
            error_type = gold_item['error_type']
            
            metrics_by_difficulty[difficulty].append((is_correct, confidence))
            metrics_by_domain[domain].append((is_correct, confidence))
            metrics_by_error_type[error_type].append((is_correct, confidence))
            
            # Reasoning depth: number of charges/evidence points
            charges = pred_item.get('charges', [])
            reasoning_depth = len(charges) if charges else 1
            reasoning_depth_scores.append(reasoning_depth)
            
            # Explanation quality (simple heuristic based on explanation length and keywords)
            explanation = pred.get('meta', {}).get('explanation', '')
            explanation_score = min(len(explanation.split()) / 50.0, 1.0)  # Normalize by 50 words
            explanation_quality_scores.append(explanation_score)
        
        # Compute group-wise accuracies
        group_accuracies = {}
        
        for difficulty, results in metrics_by_difficulty.items():
            accuracy = np.mean([r[0] for r in results])
            confidence = np.mean([r[1] for r in results])
            group_accuracies[f'accuracy_{difficulty}'] = accuracy
            group_accuracies[f'confidence_{difficulty}'] = confidence
        
        for domain, results in metrics_by_domain.items():
            accuracy = np.mean([r[0] for r in results])
            group_accuracies[f'accuracy_{domain}'] = accuracy
            
        for error_type, results in metrics_by_error_type.items():
            accuracy = np.mean([r[0] for r in results])
            group_accuracies[f'accuracy_{error_type}'] = accuracy
        
        return {
            **group_accuracies,
            'avg_reasoning_depth': np.mean(reasoning_depth_scores),
            'avg_explanation_quality': np.mean(explanation_quality_scores),
            'consistency_score': self._compute_consistency_score(system_name),
            'robustness_score': self._compute_robustness_score(system_name)
        }
    
    def compute_roundtable_specific_metrics(self, system_name: str) -> Dict[str, float]:
        """Metrics specific to roundtable/conversational systems"""
        predictions = self.results[system_name]
        
        conversation_lengths = []
        consensus_rates = []
        debate_quality_scores = []
        agent_participation_balance = []
        
        for pred in predictions:
            # Skip if not a roundtable system
            if pred.get('meta', {}).get('framework') != 'HEDA-RoundTable':
                continue
                
            conversation = pred.get('roundtable_conversation', [])
            conversation_lengths.append(len(conversation))
            
            # Consensus rate (how often agents agreed)
            consensus_points = len(pred.get('summary', {}).get('consensus_points', []))
            total_charges = pred.get('summary', {}).get('num_charges', 1)
            consensus_rates.append(consensus_points / max(total_charges, 1))
            
            # Debate quality
            debate_quality = pred.get('summary', {}).get('debate_quality', 0.5)
            debate_quality_scores.append(debate_quality)
            
            # Agent participation balance (how evenly did agents participate?)
            agent_turns = defaultdict(int)
            for turn in conversation:
                agent_turns[turn.get('agent', 'unknown')] += 1
            
            if len(agent_turns) > 1:
                turn_counts = list(agent_turns.values())
                # Calculate coefficient of variation (lower = more balanced)
                balance_score = 1.0 - (np.std(turn_counts) / np.mean(turn_counts))
                agent_participation_balance.append(max(balance_score, 0))
            else:
                agent_participation_balance.append(0)
        
        if not conversation_lengths:
            return {'roundtable_samples': 0}
            
        return {
            'avg_conversation_length': np.mean(conversation_lengths),
            'avg_consensus_rate': np.mean(consensus_rates),
            'avg_debate_quality': np.mean(debate_quality_scores),
            'avg_participation_balance': np.mean(agent_participation_balance),
            'roundtable_samples': len(conversation_lengths)
        }
    
    def _expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:  # Last bin includes right edge
                bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_prob[bin_mask])
                bin_weight = np.sum(bin_mask) / len(y_true)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    
    def _compute_consistency_score(self, system_name: str) -> float:
        """Measure how consistent the system is on similar samples"""
        # This would require additional data about similar samples
        # For now, return a placeholder based on confidence variance
        predictions = self.results[system_name]
        confidences = [p['prediction']['confidence'] for p in predictions]
        consistency = 1.0 - (np.std(confidences) / max(np.mean(confidences), 0.1))
        return max(consistency, 0.0)
    
    def _compute_robustness_score(self, system_name: str) -> float:
        """Measure robustness across different types of errors"""
        predictions = self.results[system_name]
        
        error_type_accuracies = defaultdict(list)
        
        for pred in predictions:
            sample_id = pred['id']
            if sample_id not in self.gold:
                continue
                
            gold_item = self.gold[sample_id]
            pred_item = pred['prediction']
            
            is_correct = gold_item['has_error'] == pred_item['has_error']
            error_type = gold_item['error_type']
            
            error_type_accuracies[error_type].append(is_correct)
        
        if len(error_type_accuracies) <= 1:
            return 0.5  # Not enough diversity to measure robustness
            
        # Robustness = 1 - coefficient of variation of accuracies across error types
        type_accs = [np.mean(accs) for accs in error_type_accuracies.values()]
        if np.mean(type_accs) == 0:
            return 0.0
            
        robustness = 1.0 - (np.std(type_accs) / np.mean(type_accs))
        return max(robustness, 0.0)
    
    def compare_systems(self) -> pd.DataFrame:
        """Generate comprehensive comparison table"""
        all_metrics = {}
        
        for system_name in self.results.keys():
            metrics = {}
            metrics.update(self.compute_basic_metrics(system_name))
            metrics.update(self.compute_advanced_metrics(system_name))
            metrics.update(self.compute_roundtable_specific_metrics(system_name))
            all_metrics[system_name] = metrics
        
        df = pd.DataFrame(all_metrics).T
        return df.round(4)
    
    def statistical_significance_test(self, system_a: str, system_b: str) -> Dict[str, Any]:
        """Test statistical significance between two systems"""
        preds_a = self.results[system_a]
        preds_b = self.results[system_b]
        
        # Match samples by ID
        common_ids = set(p['id'] for p in preds_a) & set(p['id'] for p in preds_b)
        
        results_a = {p['id']: p['prediction']['has_error'] for p in preds_a if p['id'] in common_ids}
        results_b = {p['id']: p['prediction']['has_error'] for p in preds_b if p['id'] in common_ids}
        
        # Compute accuracy for each system on common samples
        correct_a = []
        correct_b = []
        
        for sample_id in common_ids:
            if sample_id not in self.gold:
                continue
            gold_label = self.gold[sample_id]['has_error']
            correct_a.append(results_a[sample_id] == gold_label)
            correct_b.append(results_b[sample_id] == gold_label)
        
        correct_a, correct_b = np.array(correct_a), np.array(correct_b)
        
        # McNemar's test for paired binary classifications
        diff = correct_a.astype(int) - correct_b.astype(int)
        
        # Count disagreements
        a_correct_b_wrong = np.sum(diff == 1)
        a_wrong_b_correct = np.sum(diff == -1)
        
        if a_correct_b_wrong + a_wrong_b_correct < 10:
            p_value = 1.0  # Not enough disagreements for meaningful test
        else:
            # McNemar's test statistic
            chi2_stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1)**2 / (a_correct_b_wrong + a_wrong_b_correct)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        acc_a = np.mean(correct_a)
        acc_b = np.mean(correct_b)
        
        return {
            'system_a': system_a,
            'system_b': system_b,
            'accuracy_a': float(acc_a),
            'accuracy_b': float(acc_b),
            'accuracy_diff': float(acc_a - acc_b),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'common_samples': len(correct_a),
            'a_correct_b_wrong': int(a_correct_b_wrong),
            'a_wrong_b_correct': int(a_wrong_b_correct)
        }
    
    def generate_report(self, output_path: str):
        """Generate comprehensive HTML report"""
        comparison_df = self.compare_systems()
        
        html_parts = []
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>LLM Judge Evaluation Report</title>
            <style>
                body { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
                .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0; }
                .metric-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; background: #f9fafb; }
                .metric-title { font-weight: bold; color: #374151; margin-bottom: 0.5rem; }
                .metric-value { font-size: 1.5rem; font-weight: bold; color: #059669; }
                .comparison-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
                .comparison-table th, .comparison-table td { border: 1px solid #d1d5db; padding: 8px; text-align: center; }
                .comparison-table th { background-color: #f3f4f6; font-weight: bold; }
                .best-score { background-color: #d1fae5; font-weight: bold; }
                .section { margin: 2rem 0; }
                .section h2 { border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
            </style>
        </head>
        <body>
            <h1>🎯 LLM Judge Evaluation Report</h1>
            <p>Comprehensive evaluation of reasoning judgment systems</p>
        """)
        
        # System Overview
        html_parts.append('<div class="section"><h2>📊 System Overview</h2>')
        html_parts.append('<div class="metric-grid">')
        
        for system in comparison_df.index:
            basic_metrics = self.compute_basic_metrics(system)
            html_parts.append(f"""
            <div class="metric-card">
                <div class="metric-title">{system}</div>
                <div class="metric-value">{basic_metrics['accuracy']:.3f}</div>
                <div>Accuracy • {basic_metrics['sample_count']} samples</div>
                <div>F1: {basic_metrics['f1_score']:.3f} • AUC: {basic_metrics['auc']:.3f}</div>
            </div>
            """)
        
        html_parts.append('</div></div>')
        
        # Detailed Comparison Table
        html_parts.append('<div class="section"><h2>🔍 Detailed Metrics Comparison</h2>')
        
        # Create HTML table
        html_parts.append('<table class="comparison-table">')
        
        # Header
        html_parts.append('<thead><tr><th>System</th>')
        key_metrics = ['accuracy', 'f1_score', 'auc', 'ece', 'avg_reasoning_depth']
        if any('roundtable_samples' in comparison_df.columns for _ in comparison_df.index):
            key_metrics.extend(['avg_conversation_length', 'avg_debate_quality', 'avg_participation_balance'])
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                html_parts.append(f'<th>{metric.replace("_", " ").title()}</th>')
        html_parts.append('</tr></thead>')
        
        # Body
        html_parts.append('<tbody>')
        for system in comparison_df.index:
            html_parts.append(f'<tr><td><strong>{system}</strong></td>')
            for metric in key_metrics:
                if metric in comparison_df.columns:
                    value = comparison_df.loc[system, metric]
                    if pd.notna(value):
                        # Highlight best scores
                        is_best = abs(value - comparison_df[metric].max()) < 1e-6
                        cell_class = 'class="best-score"' if is_best and len(comparison_df) > 1 else ''
                        html_parts.append(f'<td {cell_class}>{value:.3f}</td>')
                    else:
                        html_parts.append('<td>—</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody></table>')
        html_parts.append('</div>')
        
        # Statistical Significance Tests
        if len(comparison_df) >= 2:
            html_parts.append('<div class="section"><h2>📈 Statistical Significance Tests</h2>')
            html_parts.append('<table class="comparison-table">')
            html_parts.append('<thead><tr><th>System A</th><th>System B</th><th>Accuracy Diff</th><th>P-value</th><th>Significant?</th></tr></thead>')
            html_parts.append('<tbody>')
            
            systems = list(comparison_df.index)
            for i in range(len(systems)):
                for j in range(i + 1, len(systems)):
                    sig_test = self.statistical_significance_test(systems[i], systems[j])
                    significant_mark = "✅ Yes" if sig_test['significant'] else "❌ No"
                    html_parts.append(f"""
                    <tr>
                        <td>{sig_test['system_a']}</td>
                        <td>{sig_test['system_b']}</td>
                        <td>{sig_test['accuracy_diff']:+.3f}</td>
                        <td>{sig_test['p_value']:.3f}</td>
                        <td>{significant_mark}</td>
                    </tr>
                    """)
            
            html_parts.append('</tbody></table></div>')
        
        # Roundtable-Specific Analysis
        roundtable_systems = [s for s in comparison_df.index if comparison_df.loc[s, 'roundtable_samples'] > 0]
        if roundtable_systems:
            html_parts.append('<div class="section"><h2>🗣️ Roundtable Discussion Analysis</h2>')
            
            for system in roundtable_systems:
                rt_metrics = self.compute_roundtable_specific_metrics(system)
                html_parts.append(f"""
                <div class="metric-card" style="margin: 1rem 0;">
                    <div class="metric-title">{system} - Roundtable Metrics</div>
                    <div class="metric-grid" style="grid-template-columns: repeat(4, 1fr);">
                        <div><strong>Avg Conversation Length:</strong> {rt_metrics.get('avg_conversation_length', 0):.1f} turns</div>
                        <div><strong>Consensus Rate:</strong> {rt_metrics.get('avg_consensus_rate', 0):.2%}</div>
                        <div><strong>Debate Quality:</strong> {rt_metrics.get('avg_debate_quality', 0):.3f}</div>
                        <div><strong>Participation Balance:</strong> {rt_metrics.get('avg_participation_balance', 0):.3f}</div>
                    </div>
                </div>
                """)
        
        # Error Analysis by Category
        html_parts.append('<div class="section"><h2>🎯 Performance by Category</h2>')
        html_parts.append('<table class="comparison-table">')
        html_parts.append('<thead><tr><th>System</th><th>Easy</th><th>Medium</th><th>Hard</th><th>Logical Errors</th><th>Factual Errors</th></tr></thead>')
        html_parts.append('<tbody>')
        
        for system in comparison_df.index:
            adv_metrics = self.compute_advanced_metrics(system)
            html_parts.append(f"""
            <tr>
                <td><strong>{system}</strong></td>
                <td>{adv_metrics.get('accuracy_easy', 0):.3f}</td>
                <td>{adv_metrics.get('accuracy_medium', 0):.3f}</td>
                <td>{adv_metrics.get('accuracy_hard', 0):.3f}</td>
                <td>{adv_metrics.get('accuracy_logical', 0):.3f}</td>
                <td>{adv_metrics.get('accuracy_factual', 0):.3f}</td>
            </tr>
            """)
        
        html_parts.append('</tbody></table></div>')
        
        # Recommendations
        html_parts.append('<div class="section"><h2>💡 Key Insights & Recommendations</h2>')
        
        best_system = comparison_df['accuracy'].idxmax()
        best_accuracy = comparison_df.loc[best_system, 'accuracy']
        
        html_parts.append(f"""
        <div class="metric-card">
            <h3>🏆 Best Performing System: {best_system}</h3>
            <ul>
                <li><strong>Overall Accuracy:</strong> {best_accuracy:.3f}</li>
                <li><strong>F1 Score:</strong> {comparison_df.loc[best_system, 'f1_score']:.3f}</li>
                <li><strong>Calibration (ECE):</strong> {comparison_df.loc[best_system, 'ece']:.3f} (lower is better)</li>
            </ul>
        </div>
        """)
        
        if roundtable_systems:
            best_rt = max(roundtable_systems, key=lambda s: comparison_df.loc[s, 'avg_debate_quality'])
            html_parts.append(f"""
            <div class="metric-card">
                <h3>🗣️ Best Roundtable System: {best_rt}</h3>
                <p>Shows superior collaborative reasoning with high debate quality and balanced agent participation.</p>
            </div>
            """)
        
        html_parts.append('</div>')
        html_parts.append('</body></html>')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        
        print(f"📊 Generated comprehensive report: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive LLM Judge Evaluation")
    parser.add_argument("--gold", required=True, help="Gold standard JSONL file")
    parser.add_argument("--systems", nargs="+", required=True, help="System result files (format: name:path)")
    parser.add_argument("--report", required=True, help="Output HTML report path")
    parser.add_argument("--json-output", help="Optional JSON output for programmatic access")
    
    args = parser.parse_args()
    
    # Initialize metrics system
    metrics = LLMJudgeMetrics(args.gold)
    
    # Add system results
    for system_spec in args.systems:
        if ':' not in system_spec:
            print(f"Warning: System spec '{system_spec}' should be 'name:path'")
            continue
        name, path = system_spec.split(':', 1)
        metrics.add_system_results(name, path)
    
    # Generate comparison
    comparison_df = metrics.compare_systems()
    print("\n📊 System Comparison Summary:")
    print(comparison_df[['accuracy', 'f1_score', 'auc', 'ece']].to_string())
    
    # Generate report
    metrics.generate_report(args.report)
    
    # Optional JSON output
    if args.json_output:
        result = {
            'comparison': comparison_df.to_dict(),
            'systems': list(metrics.results.keys()),
            'sample_count': len(metrics.gold)
        }
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    
if __name__ == "__main__":
    main()