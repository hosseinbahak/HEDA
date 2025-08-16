# app/agents/judge.py
from .base import BaseAgent
from app.config.prompts import JUDGE_PROMPT

class JudgeAgent(BaseAgent):
    def build_prompt(self, text: str, case_file: dict, reflection: dict, rebuttal: dict) -> str:
        return f"Judge: Weigh both sides and decide.\nText:\n{text}\nCaseFile:\n{case_file}\nReflection:\n{reflection}\nDefense:\n{rebuttal}"

    def run_on(self, text: str, case_file: dict, reflection: dict, rebuttal: dict):
        return self.run(prompt=self.build_prompt(text, case_file, reflection, rebuttal))
