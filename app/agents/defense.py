# app/agents/defense.py
from app.agents.base import BaseAgent
from app.config.prompts import DEFENSE_PROMPT

class DefenseAgent(BaseAgent):
    def build_prompt(self, text: str, case_file: dict, reflection: dict) -> str:
        return f"Defense: Rebut prosecutor charges with confidence.\nText:\n{text}\nCaseFile:\n{case_file}\nReflection:\n{reflection}"

    def run_on(self, text: str, case_file: dict, reflection: dict):
        return self.run(prompt=self.build_prompt(text, case_file, reflection))
