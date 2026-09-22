from collections.abc import Sequence

import pytest

from app.indexing import EmbeddingVector, FaissVectorIndex
from app.models import DocumentChunk, SourceMetadata
from app.retrieval import DenseRetriever, Retriever


class QueryEmbeddingModel:
    def __init__(self, vectors: list[EmbeddingVector]) -> None:
        self.model_name = "test-query-encoder"
        self.vectors = vectors
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        self.encoded_batches.append(list(texts))
        return self.vectors


def build_runbook_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/checkout.md",
            document_type="runbook",
            service="checkout",
            section="diagnosi",
        ),
    )


def build_checkout_index() -> FaissVectorIndex:
    index = FaissVectorIndex(dimension=3)
    index.add(
        [
            build_runbook_chunk(
                "payment-unreachable",
                "Il checkout non riesce a raggiungere il servizio payment.",
            ),
            build_runbook_chunk(
                "cart-empty",
                "Il carrello non contiene prodotti.",
            ),
            build_runbook_chunk(
                "payment-timeout",
                "Il servizio payment supera il timeout configurato.",
            ),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
    )
    return index


def test_dense_retriever_satisfies_common_retriever_contract() -> None:
    retriever = DenseRetriever(
        embedding_model=QueryEmbeddingModel([[1.0, 0.0, 0.0]]),
        vector_index=build_checkout_index(),
    )

    assert isinstance(retriever, Retriever)


def test_query_is_embedded_and_nearest_chunks_are_ranked() -> None:
    embedding_model = QueryEmbeddingModel([[1.0, 0.0, 0.0]])
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_index=build_checkout_index(),
    )

    results = retriever.retrieve(
        "  Perché il checkout non raggiunge payment?  ",
        k=2,
    )

    assert embedding_model.encoded_batches == [["Perché il checkout non raggiunge payment?"]]
    assert [result.chunk.id for result in results] == [
        "payment-unreachable",
        "payment-timeout",
    ]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8])
    assert {result.retriever for result in results} == {"dense"}


def test_top_k_larger_than_index_returns_only_available_chunks() -> None:
    retriever = DenseRetriever(
        embedding_model=QueryEmbeddingModel([[1.0, 0.0, 0.0]]),
        vector_index=build_checkout_index(),
    )

    results = retriever.retrieve("Errore durante il pagamento", k=10)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


def test_empty_index_does_not_invoke_embedding_model() -> None:
    embedding_model = QueryEmbeddingModel([[1.0, 0.0, 0.0]])
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_index=FaissVectorIndex(dimension=3),
    )

    assert retriever.retrieve("Errore nel checkout", k=3) == []
    assert embedding_model.encoded_batches == []


def test_embedding_model_must_return_one_vector_for_the_query() -> None:
    retriever = DenseRetriever(
        embedding_model=QueryEmbeddingModel([]),
        vector_index=build_checkout_index(),
    )

    with pytest.raises(ValueError, match="un solo embedding"):
        retriever.retrieve("Errore nel checkout", k=3)
