from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import anthropic

from ..models.evaluation import TestCase
from ..models.prompt import PromptVersion
from .base import BaseRunner

_PLACEHOLDER = "placeholder"


class ClaudeRunner(BaseRunner):
    """Generate responses via Anthropic Claude.

    Env var: ANTHROPIC_API_KEY (must be a real key, not 'placeholder')
    """

    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.3

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key or resolved_key.strip() == _PLACEHOLDER:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set or is 'placeholder'. "
                "Use --provider groq (free) or provide a valid Anthropic key."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    def run(
        self,
        prompt: PromptVersion,
        test_case: TestCase,
    ) -> Tuple[str, float, int, int]:
        messages = []
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

        max_tokens: int = prompt.model_params.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        temperature: float = prompt.model_params.get("temperature", self.DEFAULT_TEMPERATURE)

        t0 = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=prompt.system,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = response.content[0].text if response.content else ""
        in_tok = response.usage.input_tokens if response.usage else 0
        out_tok = response.usage.output_tokens if response.usage else 0
        return text, latency_ms, in_tok, out_tok
