"""LLMClient ABC and OpenRouterClient (OpenAI-compatible endpoint).

OpenRouter base URL: https://openrouter.ai/api/v1
Auth:               Bearer OPENROUTER_API_KEY (loaded from .env)
Model format:       "openai/gpt-4o-mini", "anthropic/claude-3.5-haiku", etc.

Fail-loud rule: if USE_HOSTED=true and OPENROUTER_API_KEY is absent,
raise ConfigError at construction time — never silently fall back.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import structlog
from dotenv import load_dotenv

from docintel.exceptions import ConfigError
from docintel.utils.config import GenerationConfig
from docintel.utils.cost_tracker import cost_tracker

# openai is imported lazily inside OpenRouterClient.__init__ so the module
# loads even when USE_HOSTED=false and the package is not installed.

log = structlog.get_logger()

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Per-model price table (USD per 1M tokens). Updated as models change.
_PRICE_PER_1M: dict[str, dict[str, float]] = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 5.0, "output": 15.0},
    "anthropic/claude-3.5-haiku": {"input": 0.80, "output": 4.0},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
}


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICE_PER_1M.get(model)
    if prices is None:
        return 0.0
    return (
        prompt_tokens * prices["input"] + completion_tokens * prices["output"]
    ) / 1_000_000


# ── ABC ───────────────────────────────────────────────────────────────────────

from docintel.generation.models import LLMResponse


class LLMClient(ABC):
    """Abstract LLM interface (DATA_MODEL §3.5)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> LLMResponse:
        """Send a user message and return a structured response."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Registry/display name for this model, e.g. 'openai/gpt-4o-mini'."""


# ── OpenRouter implementation ─────────────────────────────────────────────────


class OpenRouterClient(LLMClient):
    """Calls OpenRouter's OpenAI-compatible chat completions endpoint.

    Uses the OpenAI Python SDK with a custom base URL. The system prompt is
    set once at construction and prepended to every generate() call.
    """

    def __init__(
        self,
        config: GenerationConfig,
        system_prompt: str | None = None,
    ) -> None:
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        use_hosted = os.environ.get("USE_HOSTED", "false").lower() == "true"

        if use_hosted and not api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set but USE_HOSTED=true. "
                "Add the key to your .env file."
            )

        from openai import OpenAI

        self._client = OpenAI(
            base_url=_OPENROUTER_BASE,
            api_key=api_key or "no-key",
        )
        self._config = config
        self._system_prompt = system_prompt or ""
        log.info("llm_client_ready", model=config.model, base_url=_OPENROUTER_BASE)

    @property
    def model_name(self) -> str:
        return self._config.model

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> LLMResponse:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        log.debug("llm_generate_start", model=self._config.model)
        t0 = time.monotonic()

        response = self._client.chat.completions.create(**kwargs)

        latency_ms = round((time.monotonic() - t0) * 1000)
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost_usd = _compute_cost(self._config.model, prompt_tokens, completion_tokens)

        cost_tracker.track(
            kind="llm",
            model=self._config.model,
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        log.info(
            "llm_generate_complete",
            model=self._config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost_usd, 6),
            latency_ms=latency_ms,
        )

        return LLMResponse(
            text=text,
            model=self._config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
