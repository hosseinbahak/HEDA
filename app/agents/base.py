# app/agents/base.py
import os, json, random, re, time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

@dataclass
class LLMResponse:
    content: Dict[str, Any]
    confidence: float
    model_used: str

class LLMProvider:
    """
    Abstraction for calling OpenRouter (if configured) or using a local mock model.
    """
    def __init__(self, use_openrouter: bool = False):
        self.use_openrouter = use_openrouter
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        if use_openrouter and not self.api_key:
            # fail back to mock logic gracefully
            self.use_openrouter = False

    # inside app/agents/base.py, replace chat_json with this:
    def chat_json(self, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        import time, logging, os, json as _json
        log = logging.getLogger("heda.provider")
        t0 = time.perf_counter()

        # never send mock models to OpenRouter
        if model.startswith("mock/") or not self.use_openrouter:
            out = self._mock_reasoning(system_prompt, user_prompt)
            dt = (time.perf_counter() - t0) * 1000.0
            log.debug("mock_call", extra={"extra": {"model": model, "ms": round(dt,2)}})
            return out

        # OpenRouter call
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("APP_REFERER", "http://localhost:5000"),
            "X-Title": os.getenv("APP_NAME", "HEDA"),
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt or ""}
            ]
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60
            )
            if resp.status_code >= 400:
                try:
                    err = resp.json()
                except Exception:
                    err = {"text": resp.text}
                dt = (time.perf_counter() - t0) * 1000.0
                log.warning("openrouter_error", extra={"extra": {"model": model, "status": resp.status_code, "ms": round(dt,2), "err": err}})
                if os.getenv("FALLBACK_TO_MOCK_ON_ERROR", "true").lower() == "true":
                    out = self._mock_reasoning(system_prompt, user_prompt)
                    log.warning("fallback_to_mock", extra={"extra": {"model": model}})
                    return out
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {err}")

            content = resp.json()["choices"][0]["message"]["content"]
            try:
                out = _json.loads(content)
            except _json.JSONDecodeError:
                out = {"raw_content": content}
            dt = (time.perf_counter() - t0) * 1000.0
            log.info("openrouter_call", extra={"extra": {"model": model, "ms": round(dt,2)}})
            return out

        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000.0
            log.error("openrouter_exception", extra={"extra": {"model": model, "ms": round(dt,2), "error": str(e)}})
            if os.getenv("FALLBACK_TO_MOCK_ON_ERROR", "true").lower() == "true":
                out = self._mock_reasoning(system_prompt, user_prompt)
                log.warning("fallback_to_mock", extra={"extra": {"model": model}})
                return out
            raise


    def _mock_reasoning(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        text = user_prompt

        # --- simple detectors for the demo set ---
        # Bat & Ball trap (claim or setup)
        batball_setup = ("$1.10" in text and "$1.00" in text) or "bat and a ball cost" in text.lower()
        batball_wrong_claim = "$0.10" in text or "10 cents" in text.lower()

        # Prime-sum universal claim
        prime_sum_always_even = "sum of two prime numbers is always even" in text.lower()

        # Australia capital confusion
        aus_cap_wrong = "capital of australia is sydney" in text.lower()

        # Style marker (to keep previous behavior)
        has_therefore = "therefore" in text.lower()

        # Role routing by prompt
        if "Prosecutor" in system_prompt or "Prosecutor" in user_prompt:
            charges = []

            # Bat & Ball -> E202 (calculation error) when the wrong claim appears
            if batball_setup and batball_wrong_claim:
                charges.append({
                    "charge_id": "c_batball",
                    "error_code": "E202",
                    "severity": "high",
                    "evidence": "Claims the ball costs $0.10 in the $1.10 puzzle.",
                    "confidence": 0.95
                })

            # Prime-sum -> E103 (flawed deduction / overgeneralization)
            if prime_sum_always_even:
                charges.append({
                    "charge_id": "c_primes",
                    "error_code": "E103",
                    "severity": "medium",
                    "evidence": "Universal claim is false (2 + 3 = 5 is odd).",
                    "confidence": 0.85
                })

            # Australia capital -> E201 (factual inaccuracy)
            if aus_cap_wrong:
                charges.append({
                    "charge_id": "c_capital",
                    "error_code": "E201",
                    "severity": "high",
                    "evidence": "Says Sydney is the capital of Australia; correct is Canberra.",
                    "confidence": 0.95
                })

            # Soft stylistic poke (kept from original)
            if has_therefore and not charges:
                charges.append({
                    "charge_id": "c_style",
                    "error_code": "E103",
                    "severity": "low",
                    "evidence": "Uses 'therefore' without explicit support.",
                    "confidence": 0.6
                })

            return {"case_file": charges}

        if "Reflector" in system_prompt or "Reflector" in user_prompt:
            consensus = []
            if batball_wrong_claim: consensus.append("c_batball")
            if prime_sum_always_even: consensus.append("c_primes")
            if aus_cap_wrong: consensus.append("c_capital")
            qual = 0.75 if consensus else (0.6 if has_therefore else 0.55)
            return {"consensus_points": consensus, "confidence_in_debate_quality": qual}

        if "Defense" in system_prompt or "Defense" in user_prompt:
            rebuttals = {}
            # Hard to defend these; keep lower confidence on rebuttals so Judge can sustain.
            if batball_wrong_claim:
                rebuttals["c_batball"] = {"rebuttal_argument": "Ambiguity about pricing setup.", "confidence": 0.25}
            if prime_sum_always_even:
                rebuttals["c_primes"] = {"rebuttal_argument": "Most primes are odd so sum tends to be even.", "confidence": 0.35}
            if aus_cap_wrong:
                rebuttals["c_capital"] = {"rebuttal_argument": "Sydney is the largest city; confusion is common.", "confidence": 0.2}
            # Keep stylistic one more defensible
            if has_therefore and not (batball_wrong_claim or prime_sum_always_even or aus_cap_wrong):
                rebuttals["c_style"] = {"rebuttal_argument": "Rhetorical marker, not a logical flaw.", "confidence": 0.7}
            return {"rebuttals": rebuttals, "overall_confidence": 0.6}

        if "Judge" in system_prompt or "Judge" in user_prompt:
            # If credible charges exist, sustain; else no significant errors.
            credible = 0
            if batball_wrong_claim: credible += 1
            if prime_sum_always_even: credible += 1
            if aus_cap_wrong: credible += 1

            if credible > 0:
                conf = 0.8 + 0.05 * (credible - 1)  # 0.8..0.9 range
                return {"final_verdict": "Errors Found", "confidence": conf, "reasoning": "Sustained credible charges by Prosecutor."}
            # stylistic only -> no significant errors
            if has_therefore:
                return {"final_verdict": "No Significant Errors", "confidence": 0.65, "reasoning": "Style/phrasing issues only."}
            return {"final_verdict": "No Significant Errors", "confidence": 0.6, "reasoning": "No clear errors detected."}

        return {"raw_content": "No matching mock route."}



class BaseAgent:
    def __init__(self, name: str, model: str, system_prompt: str, provider: LLMProvider):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.provider = provider

    def run(self, **kwargs) -> LLMResponse:
        user_prompt = kwargs.get("prompt", "")
        content = self.provider.chat_json(self.model, self.system_prompt, user_prompt)
        # Guess a confidence if present
        conf = content.get("overall_confidence") or content.get("confidence") or 0.6
        return LLMResponse(content=content, confidence=float(conf), model_used=self.model)
