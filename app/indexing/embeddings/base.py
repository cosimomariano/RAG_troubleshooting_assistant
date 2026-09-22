from collections.abc import Sequence
from typing import Protocol, runtime_checkable

EmbeddingVector = list[float]


@runtime_checkable
class EmbeddingModel(Protocol):
    @property
    def model_name(self) -> str:
        """Nome del modello configurato."""
        ...

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Converte un insieme di testi in una lista di vettori densi."""
