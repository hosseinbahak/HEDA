# HEDA Round Table Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core requirements
pip install langgraph langchain-core
pip install pandas numpy scikit-learn scipy matplotlib seaborn
pip install pyyaml requests flask

# Or install all at once:
pip install -r requirements.txt
```

### 2. Requirements.txt
```
# Core framework
langgraph>=0.0.40
langchain-core>=0.1.0
pyyaml>=6.0
requests>=2.28.0
flask>=2.3.0
python-dotenv>=1.0.0

# Data processing and metrics
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Visualization (optional)
matplotlib>=3.6.0
seaborn>=0.12.0

# Development
pytest>=7.0.0
black>=23.0.0
```

### 3. Environment Setup

Create `.env` file:
```bash
# OpenRouter API (for real LLM calls)
OPENROUTER_API_KEY=your_api_key_here
USE_OPENROUTER=true

# App config
APP_NAME=HEDA-RoundTable
APP_REFERER=http://localhost:5000

# Logging
LOG_LEVEL=INFO
LOG_JSON=false
LOG_TO_FILE=false

# Tracing (optional)
TRACE_SAVE_ARTIFACTS=false
FALLBACK_TO_MOCK_ON_ERROR=true
```

## 🎯 Running Evaluations

### Single System Test
```bash
# Test HEDA RoundTable on demo dataset
python eval/runners/run_heda.py \
    --input eval/datasets/enhanced_reasoning.jsonl \
    --out results/heda_roundtable_results.jsonl \
    --mode roundtable

# Test traditional HEDA
python eval/runners/run_heda.py \
    --input eval/datasets/enhanced_reasoning.jsonl \
    --out results/heda_traditional_results.jsonl \
    --mode traditional
```

### Comprehensive Comparison
```bash
# Run all systems and generate comparison report
python eval/run_comprehensive_evaluation.py \
    --dataset eval/datasets/enhanced_reasoning.jsonl \
    --output-dir results/comprehensive_eval \
    --parallel

# This will:
# 1. Run HEDA-Traditional, HEDA-RoundTable, and baseline single-LLM judges
# 2. Generate comprehensive metrics
# 3. Create HTML comparison report
# 4. Perform statistical significance tests
```

### Custom System Comparison
```bash
# Compare only specific systems
python eval/run_comprehensive_evaluation.py \
    --dataset eval/datasets/enhanced_reasoning.jsonl \
    --output-dir results/custom_eval \
    --systems HEDA-RoundTable GPT4-Single Claude-Single
```

## 🔍 Understanding the Metrics

### Basic Metrics
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Error detection performance  
- **AUC**: Area under ROC curve
- **ECE**: Expected Calibration Error (confidence calibration)

### Advanced Metrics  
- **Performance by difficulty**: Easy/Medium/Hard samples
- **Performance by domain**: Math, Physics, Logic, etc.
- **Performance by error type**: Logical, Factual, Calculation errors
- **Robustness**: Consistency across error types
- **Consistency**: Low variance in confidence scores

### Round Table Specific Metrics
- **Conversation Length**: Average turns in discussions
- **Consensus Rate**: How often agents reach agreement
- **Debate Quality**: Overall quality of the discussion
- **Participation Balance**: How evenly agents participate

## 📊 Sample Report Structure

The comprehensive evaluation generates:

1. **System Overview Cards**: Quick performance summary
2. **Detailed Metrics Table**: All metrics side-by-side
3. **Statistical Significance Tests**: McNemar's test results
4. **Round Table Analysis**: Discussion-specific metrics
5. **Error Analysis by Category**: Performance breakdowns
6. **Key Insights & Recommendations**: Actionable findings

## 🛠️ Customization

### Adding New Systems
```python
# In eval/run_comprehensive_evaluation.py
self.systems["YourSystem"] = SystemConfig(
    name="YourSystem",
    runner_script="eval/runners/run_your_system.py", 
    extra_args={"model": "your/model"},
    description="Your system description"
)
```

### Creating Custom Datasets
```json
{
    "id": "unique_id",
    "prompt": "Reasoning text to evaluate",
    "gold_has_error": true/false,
    "error_type": "logical|factual|calculation",
    "difficulty": "easy|medium|hard",
    "domain": "mathematics|physics|logic|etc",
    "gold_confidence": 0.9,
    "explanation": "Why this is correct/incorrect"
}
```

### Extending Round Table Agents

```python
# Add new agent type to the graph
class ExpertAgent(BaseAgent):
    def build_prompt(self, text: str, context: str) -> str:
        return f"Expert analysis: {text}\n\nContext: {context}"

# In round_table_graph.py
def _expert_node(self, state: RoundTableState) -> RoundTableState:
    # Your expert agent logic
    pass
```

### Custom Error Taxonomy
```python
# In app/config/taxonomy.json
{
    "E401": "Appeal to emotion",
    "E402": "Red herring fallacy", 
    "E403": "False analogy",
    "E501": "Statistical misinterpretation",
    "E502": "Cherry picking data"
}
```

## 🔬 Advanced Features

### LangGraph Visualization
```python
# Generate graph visualization
from langgraph.graph import StateGraph

# Your roundtable graph
graph = round_table.graph
graph.get_graph().draw_mermaid_png(output_file_path="roundtable_flow.png")
```

### Custom Metrics
```python
# In eval/metrics/comprehensive_metrics.py
def compute_domain_expertise_score(self, system_name: str) -> float:
    """Custom metric for domain expertise"""
    predictions = self.results[system_name]
    domain_scores = defaultdict(list)
    
    for pred in predictions:
        domain = self.gold[pred['id']].get('domain', 'general')
        is_correct = self.gold[pred['id']]['has_error'] == pred['prediction']['has_error']
        domain_scores[domain].append(is_correct)
    
    # Calculate expertise as performance on specialized domains
    specialized_domains = ['physics', 'mathematics', 'chemistry', 'medicine']
    expertise_scores = []
    
    for domain in specialized_domains:
        if domain in domain_scores and len(domain_scores[domain]) >= 3:
            expertise_scores.append(np.mean(domain_scores[domain]))
    
    return np.mean(expertise_scores) if expertise_scores else 0.5
```

### Real-time Monitoring
```python
# Add to orchestrator for live monitoring
import wandb

class MonitoredOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wandb.init(project="heda-evaluation")
    
    def evaluate_text(self, text: str):
        result = super().evaluate_text(text)
        
        # Log metrics to Weights & Biases
        wandb.log({
            "verdict": result["summary"]["verdict"],
            "confidence": result["summary"]["confidence"],
            "num_charges": result["summary"]["num_charges"],
            "debate_quality": result["summary"].get("debate_quality", 0)
        })
        
        return result
```

## 📈 Performance Optimization

### Parallel Processing
```bash
# Use multiple workers for large datasets
export RUNNER_MAX_WORKERS=4
python eval/run_comprehensive_evaluation.py --parallel
```

### Model Caching
```python
# In app/agents/base.py
from functools import lru_cache

class LLMProvider:
    @lru_cache(maxsize=1000)
    def chat_json_cached(self, model: str, system: str, user: str):
        return self.chat_json(model, system, user)
```

### Batch Processing
```python
# Process multiple samples simultaneously
def evaluate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(self.evaluate_text, text) for text in texts]
        return [f.result() for f in futures]
```

## 🐛 Debugging & Troubleshooting

### Common Issues

1. **OpenRouter API Errors**
```bash
# Check API key and quotas
export LOG_LEVEL=DEBUG
python your_script.py

# Fallback to mock mode
export USE_OPENROUTER=false
export FALLBACK_TO_MOCK_ON_ERROR=true
```

2. **LangGraph State Issues**
```python
# Debug state transitions
import logging
logging.getLogger("langgraph").setLevel(logging.DEBUG)

# Inspect intermediate states
def debug_node(self, state):
    print(f"State at {self.__class__.__name__}: {state}")
    return state
```

3. **Memory Issues with Large Datasets**
```python
# Process in chunks
def process_dataset_chunks(dataset_path: str, chunk_size: int = 100):
    with open(dataset_path) as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
```

### Logging Configuration
```python
# Enhanced logging setup
import structlog

logger = structlog.get_logger("heda")

# In your evaluation code
logger.info("evaluation_started", 
    sample_id=sample_id,
    system=system_name,
    text_length=len(text)
)
```

## 📊 Example Output

### Sample Metrics JSON
```json
{
  "HEDA-RoundTable": {
    "accuracy": 0.847,
    "f1_score": 0.823,
    "auc": 0.891,
    "ece": 0.089,
    "avg_conversation_length": 6.2,
    "avg_debate_quality": 0.785,
    "accuracy_hard": 0.723,
    "accuracy_logical": 0.856
  },
  "GPT4-Single": {
    "accuracy": 0.791,
    "f1_score": 0.769,
    "auc": 0.834,
    "ece": 0.124,
    "accuracy_hard": 0.634,
    "accuracy_logical": 0.798
  }
}
```

### Sample Statistical Test
```json
{
  "system_a": "HEDA-RoundTable",
  "system_b": "GPT4-Single", 
  "accuracy_diff": 0.056,
  "p_value": 0.023,
  "significant": true,
  "common_samples": 200
}
```

## 🔄 Continuous Evaluation Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/evaluation.yml
name: HEDA Evaluation Pipeline

on:
  push:
    paths: ['app/**', 'eval/**']
  
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run evaluation
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        USE_OPENROUTER: true
      run: |
        python eval/run_comprehensive_evaluation.py \
          --dataset eval/datasets/enhanced_reasoning.jsonl \
          --output-dir results/ci_eval
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-results
        path: results/ci_eval/
```

## 📚 Further Reading

### Academic References
- **Multi-Agent Debate**: "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (Du et al., 2023)
- **LangGraph Framework**: LangChain documentation on graph-based workflows
- **Judge Evaluation**: "JudgeLM: Fine-tuned Large Language Models are Scalable Judges" (Zhu et al., 2023)

### System Architecture Papers
- **Constitutional AI**: For understanding principled AI evaluation
- **Tree of Thoughts**: For multi-step reasoning evaluation
- **Self-Consistency**: For confidence calibration in LLMs

## 🤝 Contributing

### Adding New Evaluation Domains
1. Extend the dataset with domain-specific samples
2. Add domain expertise metrics
3. Create specialized agent prompts
4. Update the comprehensive metrics system

### Improving Round Table Dynamics
1. Add more sophisticated moderator logic
2. Implement agent personality traits
3. Add memory systems for longer discussions
4. Create adaptive speaking order based on performance

---

## 🎉 You're Ready!

Your HEDA Round Table system is now set up with:
- ✅ LangGraph-based conversational evaluation
- ✅ Comprehensive metrics comparing multiple systems  
- ✅ Statistical significance testing
- ✅ Rich HTML reports with insights
- ✅ Extensible framework for new agents and metrics

Start with the demo dataset, then expand to your own reasoning evaluation challenges!