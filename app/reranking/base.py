from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from app.models import RetrievalResult

@runtime_checkable
class Reranker(Protocol):
    """Classe astratta per il riordino di candidati a partire dalla query"""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]: ...