from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ..models.evaluation import TestCase
from ..models.prompt import PromptVersion


class BaseRunner(ABC):
    """All LLM provider runners implement this interface.

    run() returns: (response_text, latency_ms, input_tokens, output_tokens)
    """

    @abstractmethod
    def run(
        self,
        prompt: PromptVersion,
        test_case: TestCase,
    ) -> Tuple[str, float, int, int]:
        ...
