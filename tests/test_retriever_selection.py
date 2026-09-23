import pytest
from app.models import RetrievalResult
from app.retrieval import RetrievalMode, Retriever, RetrieverSelector

class NamedRetriever:
    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return []

def build_selector() -> tuple[RetrieverSelector, NamedRetriever, NamedRetriever]:
    dense_retriever = NamedRetriever("dense")
    sparse_retriever = NamedRetriever("sparse")
    selector = RetrieverSelector(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    return selector, dense_retriever, sparse_retriever

def test_selector_returns_dense_retriever() -> None:
    selector, dense_retriever, _ = build_selector()

    selected_retriever = selector.select(RetrievalMode.DENSE)

    assert isinstance(selected_retriever, Retriever)
    assert selected_retriever is dense_retriever

def test_selector_returns_sparse_retriever() -> None:
    selector, _, sparse_retriever = build_selector()

    selected_retriever = selector.select("sparse")

    assert selected_retriever is sparse_retriever

def test_selector_normalizes_configuration_value() -> None:
    selector, dense_retriever, _ = build_selector()

    selected_retriever = selector.select("  DENSE  ")

    assert selected_retriever is dense_retriever

def test_selector_rejects_unsupported_mode() -> None:
    selector, _, _ = build_selector()

    with pytest.raises(ValueError, match="non supportata"):
        selector.select("hybrid")