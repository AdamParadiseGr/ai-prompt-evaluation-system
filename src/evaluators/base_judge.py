from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models.evaluation import ScoreDimension, TestCase


class BaseJudge(ABC):
    @abstractmethod
    def evaluate(self, test_case: TestCase, response: str) -> List[ScoreDimension]:
        """Returns semantic score dimensions for one (test_case, response) pair."""
