# app/agents/reflector.py
from .base import BaseAgent
from app.config.prompts import REFLECTOR_PROMPT

class ReflectorAgent(BaseAgent):
    def build_prompt(self, text: str, prosecutor_response: dict) -> str:
        return f"Reflector: Review prosecutor claims vs text and flag weak points.\nText:\n{text}\nProsecutor:\n{prosecutor_response}"

    def run_on(self, text: str, prosecutor_response: dict):
        return self.run(prompt=self.build_prompt(text, prosecutor_response))
