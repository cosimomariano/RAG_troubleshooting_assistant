from typing import Protocol, runtime_checkable

from app.models import LLMGenerationResult


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...

    def generateWithMetrics(self, prompt: str) -> LLMGenerationResult: ...
