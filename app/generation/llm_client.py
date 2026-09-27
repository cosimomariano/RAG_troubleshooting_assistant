from typing import Protocol, runtime_checkable

from app.models import LLMGenerationResult


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@runtime_checkable
class MeasuredLLMClient(LLMClient, Protocol):
    """Client che restituisce anche le metriche comunicate dal provider."""

    def generate_with_metrics(self, prompt: str) -> LLMGenerationResult: ...
