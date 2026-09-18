from typing import Protocol, runtime_checkable
from app.models import RetrievalResult

@runtime_checkable
class Retriever(Protocol):
    """Recupera i chunk rilevanti per una query testuale."""

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]: ...