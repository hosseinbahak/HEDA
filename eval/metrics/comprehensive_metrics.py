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
    
    def _parse_prediction(self, prediction):
        """Parse prediction from various formats"""
        if isinstance(prediction, bool):
            return prediction
        elif isinstance(prediction, str):
            # Try to detect error mentions in text
            error_keywords = ['error', 'bug', 'issue', 'problem', 'incorrect', 'wrong', 'mistake']
            return any(keyword in prediction.lower() for keyword in error_keywords)
        elif isinstance(prediction, dict):
            return prediction.get('has_error', False)
        return False

    def _parse_response(self, response):
        """Parse HEDA response format"""
        if isinstance(response, dict):
            # Check for HEDA's round-table format
            if 'final_answer' in response:
                return self._parse_prediction(response['final_answer'])
            elif 'consensus' in response:
                return self._parse_prediction(response['consensus'])
            elif 'decision' in response:
                return self._parse_prediction(response['decision'])
            elif 'analysis' in response:
                return self._parse_prediction(response['analysis'])
        return self._parse_prediction(response)
    
    def _extract_confidence(self, pred_data):
        """Extract confidence score from various formats"""
        if isinstance(pred_data, dict):
            # Direct confidence field
            if 'confidence' in pred_data:
                return float(pred_data['confidence'])
            # Check nested structures
            if 'prediction' in pred_data and isinstance(pred_data['prediction'], dict):
                if 'confidence' in pred_data['prediction']:
                    return float(pred_data['prediction']['confidence'])
            # Check response field
            if 'response' in pred_data and isinstance(pred_data['response'], dict):
                if 'confidence' in pred_data['response']:
                    return float(pred_data['response']['confidence'])
        # Default confidence
        return 0.5
        
    def compute_basic_metrics(self, system_name: str) -> Dict[str, float]:
        """Compute basic metrics for a system"""
        predictions = self.results.get(system_name, [])
        if not predictions:
            return {}
        
        y_true = []
        y_pred = []
        y_prob = []  # Initialize y_prob list
        
        for pred in predictions:
            # Get ground truth
            sample_id = pred.get('id', '')
            if sample_id in self.gold:
                y_true.append(self.gold[sample_id]['has_error'])
            elif 'ground_truth' in pred:
                y_true.append(pred['ground_truth'].get('has_error', False))
            else:
                y_true.append(False)  # Default if no ground truth
            
            # Parse prediction and confidence
            pred_value = False
            conf_value = 0.5
            
            # Check multiple possible locations for prediction
            if 'prediction' in pred:
                if isinstance(pred['prediction'], dict):
                    pred_value = pred['prediction'].get('has_error', False)
                    conf_value = pred['prediction'].get('confidence', 0.5)
                else:
                    pred_value = self._parse_prediction(pred['prediction'])
                    conf_value = 0.5
            elif 'response' in pred:
                pred_value = self._parse_response(pred['response'])
                conf_value = self._extract_confidence(pred['response'])
            elif 'output' in pred:
                pred_value = self._parse_response(pred['output'])
                conf_value = 0.5
            
            y_pred.append(pred_value)
            y_prob.append(conf_value)
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=True, zero_division=0
        )
        
        # Confidence calibration (Expected Calibration Error)
        ece = self._expected_calibration_error(y_true, y_prob)
        
        # Area under ROC curve
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = 0.5  # Default for constant predictions
        except (ValueError, TypeError):
            auc = 0.5
        
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
            sample_id = pred.get('id', '')
            if sample_id not in self.gold:
                continue
                
            gold_item = self.gold[sample_id]
            
            # Parse prediction
            pred_has_error = False
            if 'prediction' in pred and isinstance(pred['prediction'], dict):
                pred_has_error = pred['prediction'].get('has_error', False)
                confidence = pred['prediction'].get('confidence', 0.5)
            elif 'response' in pred:
                pred_has_error = self._parse_response(pred['response'])
                confidence = self._extract_confidence(pred['response'])
            else:
                confidence = 0.5
            
            is_correct = gold_item['has_error'] == pred_has_error
            
            # Group metrics
            difficulty = gold_item['difficulty']
            domain = gold_item['domain'] 
            error_type = gold_item['error_type']
            
            metrics_by_difficulty[difficulty].append((is_correct, confidence))
            metrics_by_domain[domain].append((is_correct, confidence))
            metrics_by_error_type[error_type].append((is_correct, confidence))
            
            # Reasoning depth
            if 'prediction' in pred and isinstance(pred['prediction'], dict):
                charges = pred['prediction'].get('charges', [])
                reasoning_depth = len(charges) if charges else 1
            else:
                reasoning_depth = 1
            reasoning_depth_scores.append(reasoning_depth)
            
            # Explanation quality
            explanation = pred.get('meta', {}).get('explanation', '')
            if not explanation and 'prediction' in pred and isinstance(pred['prediction'], dict):
                explanation = pred['prediction'].get('explanation', '')
            explanation_score = min(len(str(explanation).split()) / 50.0, 1.0)
            explanation_quality_scores.append(explanation_score)
        
        # Compute group-wise accuracies
        group_accuracies = {}
        
        for difficulty, results in metrics_by_difficulty.items():
            if results:
                accuracy = np.mean([r[0] for r in results])
                confidence = np.mean([r[1] for r in results])
                group_accuracies[f'accuracy_{difficulty}'] = accuracy
                group_accuracies[f'confidence_{difficulty}'] = confidence
        
        for domain, results in metrics_by_domain.items():
            if results:
                accuracy = np.mean([r[0] for r in results])
                group_accuracies[f'accuracy_{domain}'] = accuracy
            
        for error_type, results in metrics_by_error_type.items():
            if results:
                accuracy = np.mean([r[0] for r in results])
                group_accuracies[f'accuracy_{error_type}'] = accuracy
        
        return {
            **group_accuracies,
            'avg_reasoning_depth': np.mean(reasoning_depth_scores) if reasoning_depth_scores else 0,
            'avg_explanation_quality': np.mean(explanation_quality_scores) if explanation_quality_scores else 0,
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
            # Check if this is a roundtable system result
            is_roundtable = (
                pred.get('meta', {}).get('framework') == 'HEDA-RoundTable' or
                'roundtable_conversation' in pred or
                'roundtable' in system_name.lower()
            )
            
            if not is_roundtable:
                continue
                
            conversation = pred.get('roundtable_conversation', [])
            if conversation:
                conversation_lengths.append(len(conversation))
            
            # Consensus rate
            summary = pred.get('summary', {})
            consensus_points = len(summary.get('consensus_points', []))
            total_charges = summary.get('num_charges', 1)
            if total_charges > 0:
                consensus_rates.append(consensus_points / total_charges)
            
            # Debate quality
            debate_quality = summary.get('debate_quality', 0.5)
            debate_quality_scores.append(debate_quality)
            
            # Agent participation balance
            agent_turns = defaultdict(int)
            for turn in conversation:
                agent_turns[turn.get('agent', 'unknown')] += 1
            
            if len(agent_turns) > 1:
                turn_counts = list(agent_turns.values())
                balance_score = 1.0 - (np.std(turn_counts) / (np.mean(turn_counts) + 1e-6))
                agent_participation_balance.append(max(balance_score, 0))
            else:
                agent_participation_balance.append(0)
        
        if not conversation_lengths:
            return {'roundtable_samples': 0}
            
        return {
            'avg_conversation_length': np.mean(conversation_lengths),
            'avg_consensus_rate': np.mean(consensus_rates) if consensus_rates else 0,
            'avg_debate_quality': np.mean(debate_quality_scores) if debate_quality_scores else 0,
            'avg_participation_balance': np.mean(agent_participation_balance) if agent_participation_balance else 0,
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
        predictions = self.results[system_name]
        confidences = []
        
        for pred in predictions:
            if 'prediction' in pred and isinstance(pred['prediction'], dict):
                confidences.append(pred['prediction'].get('confidence', 0.5))
            else:
                confidences.append(self._extract_confidence(pred))
        
        if not confidences:
            return 0.5
            
        mean_conf = np.mean(confidences)
        if mean_conf == 0:
            return 0.0
            
        consistency = 1.0 - (np.std(confidences) / max(mean_conf, 0.1))
        return max(min(consistency, 1.0), 0.0)
    
    def _compute_robustness_score(self, system_name: str) -> float:
        """Measure robustness across different types of errors"""
        predictions = self.results[system_name]
        
        error_type_accuracies = defaultdict(list)
        
        for pred in predictions:
            sample_id = pred.get('id', '')
            if sample_id not in self.gold:
                continue
                
            gold_item = self.gold[sample_id]
            
            # Parse prediction
            if 'prediction' in pred and isinstance(pred['prediction'], dict):
                pred_has_error = pred['prediction'].get('has_error', False)
            elif 'response' in pred:
                pred_has_error = self._parse_response(pred['response'])
            else:
                pred_has_error = False
            
            is_correct = gold_item['has_error'] == pred_has_error
            error_type = gold_item['error_type']
            
            error_type_accuracies[error_type].append(is_correct)
        
        if len(error_type_accuracies) <= 1:
            return 0.5
            
        # Robustness = 1 - coefficient of variation
        type_accs = [np.mean(accs) for accs in error_type_accuracies.values() if accs]
        if not type_accs or np.mean(type_accs) == 0:
            return 0.0
            
        robustness = 1.0 - (np.std(type_accs) / np.mean(type_accs))
        return max(min(robustness, 1.0), 0.0)
    
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
        ids_a = {p.get('id', i): i for i, p in enumerate(preds_a)}
        ids_b = {p.get('id', i): i for i, p in enumerate(preds_b)}
        common_ids = set(ids_a.keys()) & set(ids_b.keys())
        
        # Compute accuracy for each system on common samples
        correct_a = []
        correct_b = []
        
        for sample_id in common_ids:
            if sample_id not in self.gold:
                continue
                
            gold_label = self.gold[sample_id]['has_error']
            
            # Get prediction A
            pred_a = preds_a[ids_a[sample_id]]
            if 'prediction' in pred_a and isinstance(pred_a['prediction'], dict):
                pred_a_value = pred_a['prediction'].get('has_error', False)
            elif 'response' in pred_a:
                pred_a_value = self._parse_response(pred_a['response'])
            else:
                pred_a_value = False
                
            # Get prediction B
            pred_b = preds_b[ids_b[sample_id]]
            if 'prediction' in pred_b and isinstance(pred_b['prediction'], dict):
                pred_b_value = pred_b['prediction'].get('has_error', False)
            elif 'response' in pred_b:
                pred_b_value = self._parse_response(pred_b['response'])
            else:
                pred_b_value = False
            
            correct_a.append(pred_a_value == gold_label)
            correct_b.append(pred_b_value == gold_label)
        
        if not correct_a:
            return {
                'system_a': system_a,
                'system_b': system_b,
                'accuracy_a': 0,
                'accuracy_b': 0,
                'accuracy_diff': 0,
                'p_value': 1.0,
                'significant': False,
                'common_samples': 0,
                'a_correct_b_wrong': 0,
                'a_wrong_b_correct': 0
            }
        
        correct_a, correct_b = np.array(correct_a), np.array(correct_b)
        
        # McNemar's test
        diff = correct_a.astype(int) - correct_b.astype(int)
        a_correct_b_wrong = np.sum(diff == 1)
        a_wrong_b_correct = np.sum(diff == -1)
        
        if a_correct_b_wrong + a_wrong_b_correct < 10:
            p_value = 1.0
        else:
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
        html_parts.append('<table class="comparison-table">')
        html_parts.append('<thead><tr><th>System</th>')
        
        key_metrics = ['accuracy', 'f1_score', 'auc', 'ece', 'avg_reasoning_depth']
        if 'roundtable_samples' in comparison_df.columns and any(comparison_df['roundtable_samples'] > 0):
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
        
        # Best Performing System
        best_system = comparison_df['accuracy'].idxmax()
        best_accuracy = comparison_df.loc[best_system, 'accuracy']
        
        html_parts.append(f"""
        <div class="section">
            <h2>💡 Key Insights</h2>
            <div class="metric-card">
                <h3>🏆 Best Performing System: {best_system}</h3>
                <ul>
                    <li><strong>Overall Accuracy:</strong> {best_accuracy:.3f}</li>
                    <li><strong>F1 Score:</strong> {comparison_df.loc[best_system, 'f1_score']:.3f}</li>
                    <li><strong>Calibration (ECE):</strong> {comparison_df.loc[best_system, 'ece']:.3f} (lower is better)</li>
                </ul>
            </div>
        </div>
        """)
        
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