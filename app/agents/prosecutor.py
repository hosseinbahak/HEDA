from app.agents.base import BaseAgent
from app.config.prompts import PROSECUTOR_PROMPT
import os, json, logging

log = logging.getLogger("heda.agent.prosecutor")

class ProsecutorAgent(BaseAgent):
    def build_prompt(self, text: str, context: str = "") -> str:
        return f"Prosecutor: Analyze text for reasoning errors using taxonomy.\nContext: {context}\nText:\n{text}"

    def run_on(self, text: str, context: str = ""):
        prompt = self.build_prompt(text, context)
        # save prompt if trace saving is on
        if os.getenv("TRACE_SAVE_ARTIFACTS", "false").lower() == "true":
            os.makedirs("results/prompts", exist_ok=True)
            with open("results/prompts/last_prosecutor.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
        log.debug("agent_prompt_built", extra={"extra": {"chars": len(prompt)}})
        return self.run(prompt=prompt)
