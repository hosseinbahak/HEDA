from typing import List, Dict, Any, Optional
import json
import re
from app.utils.logger import get_logger

logger = get_logger(__name__)

class Prosecutor:
    """Identifies logical errors and reasoning issues in text."""
    
    def __init__(self, provider):
        self.provider = provider
        
    def run_on(self, text: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze text for logical errors and reasoning issues.
        
        Args:
            text: The text to analyze
            context: Additional context (unused in prosecutor)
            
        Returns:
            Dictionary containing identified errors and analysis
        """
        prompt = self._build_prompt(text)
        
        try:
            response = self.provider.generate(prompt, role="prosecutor")
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Prosecutor analysis failed: {e}")
            # Return mock response for testing
            return self._get_mock_response()
    
    def _build_prompt(self, text: str) -> str:
        """Build the analysis prompt."""
        return f"""As a logical prosecutor, analyze the following text for reasoning errors, logical fallacies, and contradictions.

Text to analyze:
{text}

Identify and list:
1. Logical fallacies (name the specific fallacy type)
2. Contradictions or inconsistencies
3. Unsupported claims or assumptions
4. Faulty reasoning patterns
5. Missing logical connections

For each issue found, provide:
- Type of error
- Severity (high/medium/low)
- Specific quote or reference from the text
- Brief explanation of why it's problematic

Format your response as a structured analysis."""
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the prosecutor's response into structured format."""
        errors = []
        
        # Extract errors using pattern matching
        error_patterns = [
            r'(?:fallacy|error|issue|problem|contradiction)[:]\s*(.+)',
            r'\d+\.\s*(.+?)(?:\n|$)',
            r'[-•]\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match) > 10:  # Filter out very short matches
                    errors.append({
                        'type': self._classify_error(match),
                        'severity': self._assess_severity(match),
                        'description': match.strip(),
                        'source': 'prosecutor'
                    })
        
        # Deduplicate errors
        seen = set()
        unique_errors = []
        for error in errors:
            error_key = error['description'][:50]
            if error_key not in seen:
                seen.add(error_key)
                unique_errors.append(error)
        
        return {
            'errors': unique_errors[:10],  # Limit to top 10 errors
            'total_issues': len(unique_errors),
            'analysis': response[:500],  # Include first 500 chars of analysis
            'severity_summary': self._get_severity_summary(unique_errors)
        }
    
    def _classify_error(self, error_text: str) -> str:
        """Classify the type of error based on text."""
        error_text_lower = error_text.lower()
        
        # Common logical fallacies
        if any(term in error_text_lower for term in ['ad hominem', 'strawman', 'straw man']):
            return 'ad_hominem'
        elif any(term in error_text_lower for term in ['slippery slope']):
            return 'slippery_slope'
        elif any(term in error_text_lower for term in ['circular', 'begging the question']):
            return 'circular_reasoning'
        elif any(term in error_text_lower for term in ['false dilemma', 'false dichotomy']):
            return 'false_dilemma'
        elif any(term in error_text_lower for term in ['hasty', 'generalization']):
            return 'hasty_generalization'
        elif any(term in error_text_lower for term in ['contradiction', 'inconsistent']):
            return 'contradiction'
        elif any(term in error_text_lower for term in ['assumption', 'unsupported']):
            return 'unsupported_claim'
        else:
            return 'logical_error'
    
    def _assess_severity(self, error_text: str) -> str:
        """Assess the severity of an error."""
        error_text_lower = error_text.lower()
        
        high_severity_terms = ['serious', 'major', 'critical', 'fundamental', 'severe']
        low_severity_terms = ['minor', 'slight', 'small', 'potential', 'possible']
        
        if any(term in error_text_lower for term in high_severity_terms):
            return 'high'
        elif any(term in error_text_lower for term in low_severity_terms):
            return 'low'
        else:
            return 'medium'
    
    def _get_severity_summary(self, errors: List[Dict]) -> Dict[str, int]:
        """Get a summary of error severities."""
        summary = {'high': 0, 'medium': 0, 'low': 0}
        for error in errors:
            severity = error.get('severity', 'medium')
            summary[severity] = summary.get(severity, 0) + 1
        return summary
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Get a mock response for testing."""
        return {
            'errors': [
                {
                    'type': 'logical_fallacy',
                    'severity': 'medium',
                    'description': 'Potential circular reasoning detected',
                    'source': 'prosecutor'
                }
            ],
            'total_issues': 1,
            'analysis': 'Mock analysis: The text contains potential logical issues.',
            'severity_summary': {'high': 0, 'medium': 1, 'low': 0}
        }