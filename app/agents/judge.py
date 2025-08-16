import json
from typing import Dict, Any, List, Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

class Judge:
    """Evaluates arguments from both prosecutor and defender."""
    
    def __init__(self, provider):
        self.provider = provider
    
    def run_on(self, text: str, prosecutor_analysis: Any, defender_analysis: Any) -> Dict[str, Any]:
        """
        Judge the arguments from both sides.
        
        Args:
            text: The original text
            prosecutor_analysis: The prosecutor's analysis
            defender_analysis: The defender's analysis
            
        Returns:
            Dictionary containing judicial evaluation
        """
        # Convert analyses to strings if they're dicts
        p_claims = self._extract_claims(prosecutor_analysis)
        d_claims = self._extract_claims(defender_analysis)
        
        prompt = self._build_prompt(text, p_claims, d_claims)
        
        try:
            response = self.provider.generate(prompt, role="judge")
            return self._parse_response(response, prosecutor_analysis, defender_analysis)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return self._get_mock_response()
    
    def _extract_claims(self, analysis: Any) -> str:
        """Extract claims from analysis."""
        if isinstance(analysis, dict):
            if 'analysis' in analysis:
                return analysis['analysis']
            elif 'overall_defense' in analysis:
                return analysis['overall_defense']
            else:
                return json.dumps(analysis, indent=2)[:500]
        return str(analysis)[:500]
    
    def _build_prompt(self, text: str, prosecutor_claims: str, defender_claims: str) -> str:
        """Build the judge prompt."""
        return f"""As an impartial judge, evaluate the arguments from both the prosecutor and defender regarding this text.

Original text:
{text}

Prosecutor's arguments:
{prosecutor_claims}

Defender's arguments:
{defender_claims}

Your task:
1. Weigh the validity of each side's arguments
2. Identify which criticisms are valid and which defenses are successful
3. Determine the overall logical quality of the original text
4. Provide a balanced ruling on the text's reasoning
5. Suggest any improvements that could strengthen the logic

Be fair and impartial. Base your judgment on logical principles and reasoning quality."""
    
    def _parse_response(self, response: str, prosecutor_analysis: Any, defender_analysis: Any) -> Dict[str, Any]:
        """Parse the judge's response."""
        return {
            'ruling': response[:500],
            'validated_errors': self._extract_validated_errors(response, prosecutor_analysis),
            'accepted_defenses': self._extract_accepted_defenses(response, defender_analysis),
            'overall_assessment': self._determine_assessment(response),
            'improvements_suggested': self._extract_improvements(response),
            'balance_score': self._calculate_balance_score(response)
        }
    
    def _extract_validated_errors(self, response: str, prosecutor_analysis: Any) -> List[Dict]:
        """Extract which prosecutor claims were validated."""
        validated = []
        
        if isinstance(prosecutor_analysis, dict) and 'errors' in prosecutor_analysis:
            # Simple heuristic: if judge mentions similar terms, consider validated
            response_lower = response.lower()
            for error in prosecutor_analysis['errors'][:5]:
                error_desc = error.get('description', '').lower()
                # Check if key terms from error appear in judge's response
                if any(term in response_lower for term in error_desc.split()[:3]):
                    validated.append({
                        'type': error.get('type'),
                        'severity': error.get('severity'),
                        'validated': True
                    })
        
        return validated
    
    def _extract_accepted_defenses(self, response: str, defender_analysis: Any) -> List[str]:
        """Extract which defenses were accepted."""
        accepted = []
        
        acceptance_terms = ['valid point', 'correct', 'agree', 'defense holds', 'reasonable']
        response_lower = response.lower()
        
        for term in acceptance_terms:
            if term in response_lower:
                # Extract the sentence containing the acceptance
                sentences = response.split('.')
                for sentence in sentences:
                    if term in sentence.lower():
                        accepted.append(sentence.strip())
                        break
        
        return accepted[:3]
    
    def _determine_assessment(self, response: str) -> str:
        """Determine overall assessment from judge's response."""
        response_lower = response.lower()
        
        positive_indicators = ['sound', 'valid', 'strong', 'logical', 'coherent']
        negative_indicators = ['flawed', 'weak', 'illogical', 'poor', 'problematic']
        
        positive_count = sum(1 for ind in positive_indicators if ind in response_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in response_lower)
        
        if positive_count > negative_count:
            return 'mostly_sound'
        elif negative_count > positive_count:
            return 'logically_flawed'
        else:
            return 'mixed_quality'
    
    def _extract_improvements(self, response: str) -> List[str]:
        """Extract suggested improvements."""
        improvements = []
        
        improvement_indicators = ['should', 'could', 'improve', 'suggest', 'recommend', 'better']
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in improvement_indicators):
                improvements.append(sentence.strip())
        
        return improvements[:3]
    
    def _calculate_balance_score(self, response: str) -> float:
        """Calculate a balance score between prosecutor and defender."""
        # Simple heuristic based on language used
        prosecutor_favor = ['prosecutor correct', 'valid criticism', 'error confirmed']
        defender_favor = ['defender correct', 'defense holds', 'criticism unfounded']
        
        response_lower = response.lower()
        p_score = sum(1 for term in prosecutor_favor if term in response_lower)
        d_score = sum(1 for term in defender_favor if term in response_lower)
        
        if p_score + d_score == 0:
            return 0.5  # Neutral
        
        return d_score / (p_score + d_score)  # 0 = prosecutor wins, 1 = defender wins
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Get a mock response for testing."""
        return {
            'ruling': 'Mock ruling: The text contains both valid reasoning and some logical issues.',
            'validated_errors': [{'type': 'logical_fallacy', 'severity': 'medium', 'validated': True}],
            'accepted_defenses': ['Some contextual defenses are valid'],
            'overall_assessment': 'mixed_quality',
            'improvements_suggested': ['Clarify assumptions', 'Strengthen logical connections'],
            'balance_score': 0.5
        }