from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    user: str
    assistant: str


class PromptVersion(BaseModel):
    name: str
    version: str
    system: str
    user_template: str
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)
    model_params: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "PromptVersion":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
