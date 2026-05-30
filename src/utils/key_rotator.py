"""Transparent Groq API key rotator.

Reads from GROQ_API_KEYS=key1,key2,key3 (comma-separated) in .env,
falls back to GROQ_API_KEY for single-key setups.

On RateLimitError automatically switches to next key — callers see nothing.
Shared module-level instance so runner and judge rotate in sync.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI, RateLimitError
from rich.console import Console

_BASE_URL = "https://api.groq.com/openai/v1"
_console = Console()
_instance: Optional["GroqKeyRotator"] = None


def get_rotator(api_key: Optional[str] = None) -> "GroqKeyRotator":
    """Return the shared module-level rotator (created once)."""
    global _instance
    if _instance is None:
        _instance = GroqKeyRotator.from_env(extra_key=api_key)
    return _instance


class GroqKeyRotator:
    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise EnvironmentError(
                "No Groq API keys found.\n"
                "Add to .env:  GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3\n"
                "or:           GROQ_API_KEY=gsk_single_key"
            )
        self.keys = keys
        self._idx = 0
        _console.print(
            f"  [dim]Groq rotator: {len(keys)} key(s) loaded[/dim]"
        )

    @classmethod
    def from_env(cls, extra_key: Optional[str] = None) -> "GroqKeyRotator":
        keys: list[str] = []

        # GROQ_API_KEYS=key1,key2,key3  (comma-separated)
        multi = os.environ.get("GROQ_API_KEYS", "").strip()
        if multi:
            keys = [k.strip() for k in multi.split(",") if k.strip()]

        # GROQ_API_KEY_1, GROQ_API_KEY_2, ...  (numbered)
        if not keys:
            i = 1
            while True:
                k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
                if not k:
                    break
                keys.append(k)
                i += 1

        # GROQ_API_KEY=key  (single-key fallback)
        if not keys:
            single = os.environ.get("GROQ_API_KEY", "").strip()
            if single:
                keys = [single]

        # Key passed explicitly from CLI (prepend, highest priority)
        if extra_key and extra_key not in keys:
            keys.insert(0, extra_key)

        return cls(keys)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Chat completion with automatic key rotation on rate limit."""
        tried: set[int] = set()

        while len(tried) < len(self.keys):
            tried.add(self._idx)
            try:
                client = OpenAI(
                    api_key=self.keys[self._idx],
                    base_url=_BASE_URL,
                )
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            except RateLimitError:
                self._rotate(tried)

        raise RuntimeError(
            "All Groq API keys hit rate limit. "
            "Add more keys to GROQ_API_KEYS in .env"
        )

    @property
    def active_key_index(self) -> int:
        return self._idx + 1  # 1-based for display

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _rotate(self, tried: set[int]) -> None:
        old = self._idx
        for i in range(1, len(self.keys) + 1):
            candidate = (self._idx + i) % len(self.keys)
            if candidate not in tried:
                self._idx = candidate
                _console.print(
                    f"  [yellow]⟳ Groq key [{old + 1}/{len(self.keys)}] "
                    f"rate limited → key [{self._idx + 1}/{len(self.keys)}][/yellow]"
                )
                return
