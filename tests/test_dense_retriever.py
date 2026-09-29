from collections.abc import Sequence

import pytest

from app.indexing import EmbeddingVector, FaissVectorIndex
from app.models import DocumentChunk, SourceMetadata
from app.retrieval import DenseRetriever, Retriever


class QueryEmbeddingModel:
    def __init__(self, vectors: list[EmbeddingVector]) -> None:
        self.modelName = "test-query-encoder"
        self.vectors = vectors
        self.encodedBatches: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        self.encodedBatches.append(list(texts))
        return self.vectors


def buildRunbookChunk(chunkId: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunkId,
        documentId="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/checkout.md",
            documentType="runbook",
            service="checkout",
            section="diagnosi",
        ),
    )


def buildCheckoutIndex() -> FaissVectorIndex:
    index = FaissVectorIndex(dimension=3)
    index.add(
        [
            buildRunbookChunk(
                "payment-unreachable",
                "Il checkout non riesce a raggiungere il servizio payment.",
            ),
            buildRunbookChunk(
                "cart-empty",
                "Il carrello non contiene prodotti.",
            ),
            buildRunbookChunk(
                "payment-timeout",
                "Il servizio payment supera il timeout configurato.",
            ),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
    )
    return index


def testDenseRetrieverSatisfiesCommonRetrieverContract() -> None:
    retriever = DenseRetriever(
        embeddingModel=QueryEmbeddingModel([[1.0, 0.0, 0.0]]),
        vectorIndex=buildCheckoutIndex(),
    )

    assert isinstance(retriever, Retriever)


def testQueryIsEmbeddedAndNearestChunksAreRanked() -> None:
    embeddingModel = QueryEmbeddingModel([[1.0, 0.0, 0.0]])
    retriever = DenseRetriever(
        embeddingModel=embeddingModel,
        vectorIndex=buildCheckoutIndex(),
    )

    results = retriever.retrieve(
        "  Perché il checkout non raggiunge payment?  ",
        k=2,
    )

    assert embeddingModel.encodedBatches == [["Perché il checkout non raggiunge payment?"]]
    assert [result.chunk.id for result in results] == [
        "payment-unreachable",
        "payment-timeout",
    ]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8])
    assert {result.retriever for result in results} == {"dense"}


def testTopKLargerThanIndexReturnsOnlyAvailableChunks() -> None:
    retriever = DenseRetriever(
        embeddingModel=QueryEmbeddingModel([[1.0, 0.0, 0.0]]),
        vectorIndex=buildCheckoutIndex(),
    )

    results = retriever.retrieve("Errore durante il pagamento", k=10)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


def testEmptyIndexDoesNotInvokeEmbeddingModel() -> None:
    embeddingModel = QueryEmbeddingModel([[1.0, 0.0, 0.0]])
    retriever = DenseRetriever(
        embeddingModel=embeddingModel,
        vectorIndex=FaissVectorIndex(dimension=3),
    )

    assert retriever.retrieve("Errore nel checkout", k=3) == []
    assert embeddingModel.encodedBatches == []


def testEmbeddingModelMustReturnOneVectorForTheQuery() -> None:
    retriever = DenseRetriever(
        embeddingModel=QueryEmbeddingModel([]),
        vectorIndex=buildCheckoutIndex(),
    )

    with pytest.raises(ValueError, match="un solo embedding"):
        retriever.retrieve("Errore nel checkout", k=3)


@pytest.mark.parametrize(
    ("query", "topK", "expectedMessage"),
    [
        pytest.param("   ", 2, "query", id="query-vuota"),
        pytest.param("payment failure", 0, "parametro k", id="top-k-non-positivo"),
    ],
)
def testDenseRetrieverRejectsInvalidRequests(
    query: str,
    topK: int,
    expectedMessage: str,
) -> None:
    embeddingModel = QueryEmbeddingModel([[1.0, 0.0, 0.0]])
    retriever = DenseRetriever(
        embeddingModel=embeddingModel,
        vectorIndex=buildCheckoutIndex(),
    )

    with pytest.raises(ValueError, match=expectedMessage):
        retriever.retrieve(query, topK)

    assert embeddingModel.encodedBatches == []
