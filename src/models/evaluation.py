from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScoreDimension(BaseModel):
    name: str
    score: float
    reasoning: str = ""
    weight: float = 1.0


class TestCase(BaseModel):
    id: str
    question: str
    context: Optional[str] = None
    expected_topics: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class EvaluationResult(BaseModel):
    experiment_id: str
    test_case_id: str
    prompt_version: str
    prompt_name: str
    raw_response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    dimensions: List[ScoreDimension] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions)
        if total_weight == 0:
            return 0.0
        return sum(d.score * d.weight for d in self.dimensions) / total_weight
