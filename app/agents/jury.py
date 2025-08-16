from typing import Dict, Any, List
import json
from app.utils.logger import get_logger

logger = get_logger(__name__)

class Jury:
    """Provides final verdict based on all arguments."""
    
    def __init__(self, provider):
        self.provider = provider
    
    def run_on(self, text: str, prosecutor_analysis: Any, defender_analysis: Any, judge_ruling: Any) -> Dict[str, Any]:
        """
        Provide final verdict based on all arguments.
        
        Args:
            text: The original text
            prosecutor_analysis: The prosecutor's analysis
            defender_analysis: The defender's analysis
            judge_ruling: The judge's ruling
            
        Returns:
            Dictionary containing final verdict
        """
        # Extract key points from each analysis
        p_summary = self._summarize_analysis(prosecutor_analysis, 'prosecutor')
        d_summary = self._summarize_analysis(defender_analysis, 'defender')
        j_summary = self._summarize_analysis(judge_ruling, 'judge')
        
        prompt = self._build_prompt(text, p_summary, d_summary, j_summary)
        
        try:
            response = self.provider.generate(prompt, role="jury")
            return self._parse_response(response, prosecutor_analysis, defender_analysis, judge_ruling)
        except Exception as e:
            logger.error(f"Jury deliberation failed: {e}")
            return self._get_mock_response()
    
    def _summarize_analysis(self, analysis: Any, role: str) -> str:
        """Summarize an analysis for the jury."""
        if isinstance(analysis, dict):
            if role == 'prosecutor' and 'errors' in analysis:
                errors = analysis['errors'][:3]
                return f"Found {len(analysis['errors'])} issues: " + \
                       ", ".join([e.get('description', '')[:50] for e in errors])
            elif role == 'defender' and 'overall_defense' in analysis:
                return analysis['overall_defense'][:200]
            elif role == 'judge' and 'ruling' in analysis:
                return analysis['ruling'][:200]
            else:
                return json.dumps(analysis, indent=2)[:200]
        return str(analysis)[:200]
    
    def _build_prompt(self, text: str, p_summary: str, d_summary: str, j_summary: str) -> str:
        """Build the jury prompt."""
        return f"""As the jury, provide the final verdict on the logical quality of this text after considering all arguments.

Original text:
{text}

Prosecutor's case summary:
{p_summary}

Defender's case summary:
{d_summary}

Judge's ruling summary:
{j_summary}

Provide your final verdict including:
1. Overall logical quality score (1-10, where 10 is perfectly logical)
2. List of confirmed logical errors (if any)
3. List of validated strengths (if any)
4. Final recommendation (accept as logically sound, needs revision, or fundamentally flawed)
5. Key takeaways for improvement

Be decisive but fair. Your verdict is final."""
    
    def _parse_response(self, response: str, prosecutor_analysis: Any, defender_analysis: Any, judge_ruling: Any) -> Dict[str, Any]:
        """Parse the jury's verdict."""
        score = self._extract_score(response)
        
        return {
            'verdict': response[:500],
            'logical_quality_score': score,
            'confirmed_errors': self._extract_confirmed_errors(response, prosecutor_analysis),
            'validated_strengths': self._extract_strengths(response, defender_analysis),
            'final_recommendation': self._determine_recommendation(score, response),
            'improvement_priorities': self._extract_improvements(response),
            'consensus_reached': self._check_consensus(response)
        }
    
    def _extract_score(self, response: str) -> int:
        """Extract the logical quality score from response."""
        import re
        
        # Look for patterns like "8/10", "score: 7", "rate it 6", etc.
        patterns = [
            r'(\d+)\s*/\s*10',
            r'score[:\s]+(\d+)',
            r'rate[:\s]+(\d+)',
            r'quality[:\s]+(\d+)',
            r'\b([1-9]|10)\b\s*(?:out of|/)?\s*(?:10)?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        return score
                except:
                    continue
        
        # Default based on sentiment
        if 'excellent' in response.lower() or 'perfect' in response.lower():
            return 9
        elif 'good' in response.lower() or 'sound' in response.lower():
            return 7
        elif 'poor' in response.lower() or 'flawed' in response.lower():
            return 3
        else:
            return 5
    
    def _extract_confirmed_errors(self, response: str, prosecutor_analysis: Any) -> List[Dict]:
        """Extract confirmed errors from verdict."""
        confirmed = []
        
        if isinstance(prosecutor_analysis, dict) and 'errors' in prosecutor_analysis:
            for error in prosecutor_analysis['errors'][:5]:
                # Simple check if error type is mentioned in verdict
                if error.get('type', '') in response.lower():
                    confirmed.append({
                        'type': error.get('type'),
                        'severity': error.get('severity'),
                        'description': error.get('description', '')[:100]
                    })
        
        return confirmed
    
    def _extract_strengths(self, response: str, defender_analysis: Any) -> List[str]:
        """Extract validated strengths."""
        strengths = []
        
        strength_terms = ['strength', 'strong point', 'valid', 'sound', 'logical', 'coherent']
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in strength_terms):
                strengths.append(sentence.strip())
        
        return strengths[:3]
    
    def _determine_recommendation(self, score: int, response: str) -> str:
        """Determine final recommendation based on score and response."""
        if score >= 8:
            return 'accept_as_sound'
        elif score >= 5:
            return 'needs_minor_revision'
        elif score >= 3:
            return 'needs_major_revision'
        else:
            return 'fundamentally_flawed'
    
    def _extract_improvements(self, response: str) -> List[str]:
        """Extract improvement priorities."""
        improvements = []
        
        improvement_indicators = ['improve', 'fix', 'address', 'revise', 'clarify', 'strengthen']
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in improvement_indicators):
                improvements.append(sentence.strip())
        
        return improvements[:5]
    
    def _check_consensus(self, response: str) -> bool:
        """Check if consensus was reached."""
        consensus_terms = ['unanimous', 'agree', 'consensus', 'clear', 'obvious']
        return any(term in response.lower() for term in consensus_terms)
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Get a mock response for testing."""
        return {
            'verdict': 'Mock verdict: The text shows mixed logical quality with room for improvement.',
            'logical_quality_score': 6,
            'confirmed_errors': [
                {'type': 'logical_fallacy', 'severity': 'medium', 'description': 'Circular reasoning detected'}
            ],
            'validated_strengths': ['Clear argument structure'],
            'final_recommendation': 'needs_minor_revision',
            'improvement_priorities': ['Clarify assumptions', 'Strengthen evidence'],
            'consensus_reached': True
        }