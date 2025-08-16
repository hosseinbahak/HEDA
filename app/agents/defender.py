from typing import Dict, Any, List
import json
from app.utils.logger import get_logger

logger = get_logger(__name__)

class Defender:
    """Provides counterarguments and defends the logical consistency of text."""
    
    def __init__(self, provider):
        self.provider = provider
    
    def run_on(self, text: str, prosecutor_analysis: Any) -> Dict[str, Any]:
        """
        Defend the text against prosecutor's claims.
        
        Args:
            text: The original text
            prosecutor_analysis: The prosecutor's analysis (can be string or dict)
            
        Returns:
            Dictionary containing defense arguments
        """
        # Handle both string and dict prosecutor analysis
        if isinstance(prosecutor_analysis, dict):
            prosecutor_claims = prosecutor_analysis.get('analysis', str(prosecutor_analysis))
        else:
            prosecutor_claims = str(prosecutor_analysis)
        
        prompt = self._build_prompt(text, prosecutor_claims)
        
        try:
            response = self.provider.generate(prompt, role="defender")
            return self._parse_response(response, prosecutor_analysis)
        except Exception as e:
            logger.error(f"Defender analysis failed: {e}")
            return self._get_mock_response()
    
    def _build_prompt(self, text: str, prosecutor_claims: str) -> str:
        """Build the defense prompt."""
        return f"""As a logical defender, provide counterarguments to the prosecutor's claims about this text.

Original text:
{text}

Prosecutor's claims:
{prosecutor_claims}

Your task:
1. Address each criticism raised by the prosecutor
2. Provide alternative interpretations that support the text's logic
3. Identify any misunderstandings or misrepresentations in the prosecutor's analysis
4. Highlight the logical strengths of the original text
5. Explain why apparent errors might actually be valid reasoning

Be fair but thorough in your defense. Acknowledge genuine issues while defending against unfair criticisms."""
    
    def _parse_response(self, response: str, prosecutor_analysis: Any) -> Dict[str, Any]:
        """Parse the defender's response."""
        defenses = []
        
        # Extract defenses for each prosecutor claim
        if isinstance(prosecutor_analysis, dict) and 'errors' in prosecutor_analysis:
            for error in prosecutor_analysis['errors'][:5]:  # Defend against top 5 errors
                defenses.append({
                    'against_error': error.get('type', 'unknown'),
                    'defense_type': 'counterargument',
                    'argument': f"Defense against: {error.get('description', '')}",
                    'strength': 'medium'
                })
        
        return {
            'defenses': defenses,
            'overall_defense': response[:500],
            'defense_strategy': self._identify_strategy(response),
            'concessions': self._identify_concessions(response),
            'strengths_highlighted': self._identify_strengths(response)
        }
    
    def _identify_strategy(self, response: str) -> str:
        """Identify the main defense strategy used."""
        response_lower = response.lower()
        
        if 'context' in response_lower or 'understand' in response_lower:
            return 'contextual_defense'
        elif 'interpret' in response_lower or 'alternative' in response_lower:
            return 'alternative_interpretation'
        elif 'valid' in response_lower or 'justified' in response_lower:
            return 'validation'
        else:
            return 'general_defense'
    
    def _identify_concessions(self, response: str) -> List[str]:
        """Identify any concessions made by the defender."""
        concessions = []
        concession_phrases = ['however', 'admittedly', 'it is true that', 'while', 'although']
        
        for phrase in concession_phrases:
            if phrase in response.lower():
                # Extract sentence containing the phrase
                sentences = response.split('.')
                for sentence in sentences:
                    if phrase in sentence.lower():
                        concessions.append(sentence.strip())
                        break
        
        return concessions[:3]  # Return top 3 concessions
    
    def _identify_strengths(self, response: str) -> List[str]:
        """Identify strengths highlighted by the defender."""
        strengths = []
        strength_indicators = ['strong', 'valid', 'logical', 'sound', 'coherent', 'reasonable']
        
        sentences = response.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in strength_indicators):
                strengths.append(sentence.strip())
        
        return strengths[:3]  # Return top 3 strengths
    
    def _get_mock_response(self) -> Dict[str, Any]:
        """Get a mock response for testing."""
        return {
            'defenses': [
                {
                    'against_error': 'logical_fallacy',
                    'defense_type': 'counterargument',
                    'argument': 'The reasoning is contextually valid',
                    'strength': 'medium'
                }
            ],
            'overall_defense': 'Mock defense: The text\'s logic is sound when properly contextualized.',
            'defense_strategy': 'contextual_defense',
            'concessions': [],
            'strengths_highlighted': ['The argument structure is coherent']
        }