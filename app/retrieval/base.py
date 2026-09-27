from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.models import RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Recupera i chunk rilevanti per una query testuale."""

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]: ...


@dataclass(frozen=True)
class RetrievalExecution:
    """Risultati del retrieval e latenze misurate dai suoi stadi interni."""

    results: tuple[RetrievalResult, ...]
    retrieval_latency_ms: float
    reranking_latency_ms: float = 0.0


@runtime_checkable
class MeasuredRetriever(Retriever, Protocol):
    def retrieve_with_metrics(self, query: str, k: int) -> RetrievalExecution: ...
