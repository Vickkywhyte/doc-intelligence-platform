"""FakeLLMClient — deterministic LLM mock for unit tests.

Never calls the network. Returns canned responses from a queue; falls back to
a configurable default (the standard refusal phrase) when the queue is empty.
"""

from __future__ import annotations

from docintel.generation.llm_client import LLMClient
from docintel.generation.models import LLMResponse


class FakeLLMClient(LLMClient):
    """Queue-backed fake for unit tests.

    Usage:
        llm = FakeLLMClient(["Answer one [doc:1:chunk].", "Answer two."])
        resp = llm.generate("question")   # returns "Answer one [doc:1:chunk]."
        resp = llm.generate("question 2") # returns "Answer two."
        resp = llm.generate("question 3") # returns default refusal
    """

    _DEFAULT = "I don't know based on the provided documents."

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        model: str = "fake",
        default: str | None = None,
    ) -> None:
        self._queue: list[str] = list(responses or [])
        self._model = model
        self._default = default if default is not None else self._DEFAULT
        self.call_count: int = 0
        self.last_prompt: str | None = None

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = prompt
        text = self._queue.pop(0) if self._queue else self._default
        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            cost_usd=0.0,
            latency_ms=0,
        )
