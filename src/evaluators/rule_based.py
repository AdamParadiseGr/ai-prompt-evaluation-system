from __future__ import annotations

import json
import re
from typing import List

from ..models.evaluation import ScoreDimension, TestCase


_RED_FLAGS = [
    "гарантированно", "100%", "обязательно сработает",
    "я не знаю", "не уверен", "возможно неправильно",
    "i don't know", "i'm not sure", "definitely", "guaranteed",
]

_REQUIRED_JSON_KEYS = {"answer", "steps", "confidence", "requires_specialist", "category"}


class RuleBasedEvaluator:
    """Deterministic checks — no API calls, runs in < 1 ms."""

    def evaluate(
        self,
        test_case: TestCase,
        response: str,
        prompt_name: str,
    ) -> List[ScoreDimension]:
        dims: List[ScoreDimension] = []

        dims.append(ScoreDimension(
            name="length_appropriateness",
            score=self._score_length(response),
            reasoning=f"{len(response)} chars",
            weight=0.5,
        ))

        if "structured" in prompt_name.lower():
            score, reason = self._score_json(response)
            dims.append(ScoreDimension(
                name="format_compliance",
                score=score,
                reasoning=reason,
                weight=1.5,
            ))

        if test_case.expected_topics:
            score, reason = self._score_topics(response, test_case.expected_topics)
            dims.append(ScoreDimension(
                name="topic_coverage",
                score=score,
                reasoning=reason,
                weight=1.0,
            ))

        dims.append(ScoreDimension(
            name="no_red_flags",
            score=self._score_red_flags(response),
            reasoning="overconfidence / hallucination signals",
            weight=0.5,
        ))

        return dims

    @staticmethod
    def _score_length(response: str) -> float:
        n = len(response)
        if 80 <= n <= 1500:
            return 1.0
        if n < 20:
            return 0.1
        if n < 80:
            return 0.5
        return 0.75  # > 1500: mild penalty

    @staticmethod
    def _score_json(response: str) -> tuple[float, str]:
        cleaned = response.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    return 0.6, "JSON found but not root response"
                except json.JSONDecodeError:
                    pass
            return 0.0, "No valid JSON found"
        missing = _REQUIRED_JSON_KEYS - set(data.keys())
        if not missing:
            return 1.0, "All required keys present"
        return max(0.0, 1.0 - len(missing) * 0.2), f"Missing keys: {missing}"

    @staticmethod
    def _stem(word: str) -> str:
        """Naive Russian stemmer — first 6 chars covers most declensions.
        кредитная→кредит, история→истори, блокировка→блокир, фишинг→фишинг
        """
        return word[:6] if len(word) > 6 else word

    @staticmethod
    def _score_topics(response: str, topics: List[str]) -> tuple[float, str]:
        response_lower = response.lower()
        covered = []
        for topic in topics:
            # Split multi-word topics, stem each word, check any root hits
            words = [w for w in topic.lower().split() if len(w) > 3]
            roots = [RuleBasedEvaluator._stem(w) for w in words]
            if any(root in response_lower for root in roots):
                covered.append(topic)
        return len(covered) / len(topics), f"Covered {len(covered)}/{len(topics)}: {covered}"

    @staticmethod
    def _score_red_flags(response: str) -> float:
        lower = response.lower()
        found = [f for f in _RED_FLAGS if f in lower]
        return max(0.0, 1.0 - len(found) * 0.15)
