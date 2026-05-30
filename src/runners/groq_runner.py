from __future__ import annotations

import time
from typing import Optional, Tuple

from ..models.evaluation import TestCase
from ..models.prompt import PromptVersion
from ..utils.key_rotator import get_rotator
from .base import BaseRunner


class GroqRunner(BaseRunner):
    """Generate responses via Groq (free tier at console.groq.com).

    Uses the shared key rotator — no direct OpenAI client here.
    Env var: GROQ_API_KEYS (comma-separated) or GROQ_API_KEY (single).
    """

    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.3

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ) -> None:
        self.rotator = get_rotator(api_key)
        self.model = model

    def run(
        self,
        prompt: PromptVersion,
        test_case: TestCase,
    ) -> Tuple[str, float, int, int]:
        messages = [{"role": "system", "content": prompt.system}]

        for ex in prompt.few_shot_examples:
            messages.append({"role": "user", "content": ex.user})
            messages.append({"role": "assistant", "content": ex.assistant})

        messages.append({
            "role": "user",
            "content": prompt.user_template.format(
                question=test_case.question,
                context=test_case.context or "",
            ),
        })

        t0 = time.perf_counter()
        r = self.rotator.complete(
            model=self.model,
            messages=messages,
            max_tokens=prompt.model_params.get("max_tokens", self.DEFAULT_MAX_TOKENS),
            temperature=prompt.model_params.get("temperature", self.DEFAULT_TEMPERATURE),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return (
            r.choices[0].message.content or "",
            latency_ms,
            r.usage.prompt_tokens if r.usage else 0,
            r.usage.completion_tokens if r.usage else 0,
        )
