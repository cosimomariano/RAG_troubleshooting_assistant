from typing import Protocol, runtime_checkable

from app.models import DocumentChunk

SparseSearchMatch = tuple[int, float]


@runtime_checkable
class SparseIndex(Protocol):
    """Operazioni richieste da un retriever sparso."""

    def getSize(self) -> int: ...

    def getChunk(self, position: int) -> DocumentChunk: ...

    def search(self, query: str, k: int) -> list[SparseSearchMatch]: ...
