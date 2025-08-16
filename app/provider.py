import os
import json
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
from app.utils.logger import get_logger

logger = get_logger(__name__)

class Provider:
    """Handles LLM provider interactions with fallback to mock responses."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.client = None
        self.mock_mode = False
        
        if self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info("Provider initialized with OpenRouter")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter client: {e}")
                self.mock_mode = True
        else:
            logger.warning("No API key provided, using mock mode")
            self.mock_mode = True
    
    def generate(self, prompt: str, role: str = "assistant", model: str = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send
            role: The role context (prosecutor, defender, judge, jury, moderator)
            model: Optional model override
            
        Returns:
            Generated text response
        """
        if self.mock_mode:
            return self._generate_mock_response(prompt, role)
        
        try:
            model = model or "meta-llama/llama-3.2-3b-instruct:free"
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are acting as a {role} in a logical analysis discussion."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"openrouter_error: {e}")
            logger.warning("fallback_to_mock")
            return self._generate_mock_response(prompt, role)
    
    def _generate_mock_response(self, prompt: str, role: str) -> str:
        """Generate a mock response based on role."""
        mock_responses = {
            "prosecutor": """I identify the following logical issues in this text:
1. Potential circular reasoning - The argument assumes what it's trying to prove
2. Unsupported assumption - Key claims lack sufficient evidence
3. Hasty generalization - Conclusions drawn from limited examples

These issues compromise the logical integrity of the argument.""",
            
            "defender": """While the prosecutor raises some concerns, I must defend the text's logical structure:
1. The reasoning is contextually appropriate given the constraints
2. What appears as circular reasoning is actually iterative refinement
3. The generalizations are reasonable given the available data

The text maintains logical coherence despite surface-level criticisms.""",
            
            "judge": """After reviewing both arguments:
- The prosecutor correctly identifies some logical weaknesses
- The defender provides valid contextual considerations
- Both sides make reasonable points

My ruling: The text contains minor logical issues but maintains overall coherence.
Improvements needed in evidence support and assumption clarification.""",
            
            "jury": """Final verdict after considering all arguments:
Logical Quality Score: 6/10

Confirmed issues:
- Some unsupported assumptions
- Minor logical gaps

Strengths validated:
- Generally coherent structure
- Clear progression of ideas

Recommendation: Needs minor revision to strengthen logical foundations.""",
            
            "moderator": """Welcome to our logical analysis discussion. 
We will examine this text for reasoning quality, logical consistency, and argumentative strength.
Let us begin with a thorough examination of the claims and their supporting logic."""
        }
        
        # Add slight delay to simulate API call
        time.sleep(0.1)
        
        return mock_responses.get(role, f"Mock {role} response: Analyzing the provided text for logical consistency.")