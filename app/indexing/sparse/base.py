from typing import Protocol, runtime_checkable
from app.models import DocumentChunk
SparseSearchMatch = tuple[int, float]


@runtime_checkable
class SparseIndex(Protocol):
    """Classe estendibile con le operazioni di base per lo spare retriever"""

    @property
    def size(self) -> int: ...

    def get_chunk(self, position: int) -> DocumentChunk: ...
    def search(self, query: str, k: int) -> list[SparseSearchMatch]: ...
