import pytest

from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.retrieval import HybridRetriever, ReciprocalRankFusion, Retriever


class RecordingRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self.results[:k]


def buildChunk(chunkId: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunkId,
        documentId=f"document-{chunkId}",
        text=f"Contenuto diagnostico del chunk {chunkId}.",
        metadata=SourceMetadata(
            source=f"knowledge_base/{chunkId}.md",
            documentType="runbook",
        ),
    )


def buildResult(
    chunk: DocumentChunk,
    rank: int,
    score: float,
    retriever: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        rank=rank,
        score=score,
        retriever=retriever,
    )


def buildHybridRetriever() -> tuple[
    HybridRetriever,
    RecordingRetriever,
    RecordingRetriever,
]:
    payment = buildChunk("payment")
    checkout = buildChunk("checkout")
    cart = buildChunk("cart")

    sparseRetriever = RecordingRetriever(
        [
            buildResult(payment, rank=1, score=8.5, retriever="sparse"),
            buildResult(checkout, rank=2, score=4.2, retriever="sparse"),
        ]
    )
    denseRetriever = RecordingRetriever(
        [
            buildResult(checkout, rank=1, score=0.92, retriever="dense"),
            buildResult(cart, rank=2, score=0.81, retriever="dense"),
            buildResult(payment, rank=3, score=0.73, retriever="dense"),
        ]
    )
    hybridRetriever = HybridRetriever(
        sparseRetriever=sparseRetriever,
        denseRetriever=denseRetriever,
        rankFusion=ReciprocalRankFusion(),
        sparseTopK=2,
        denseTopK=3,
    )
    return hybridRetriever, sparseRetriever, denseRetriever


def testHybridRetrieverSatisfiesCommonRetrieverContract() -> None:
    hybridRetriever, _, _ = buildHybridRetriever()

    assert isinstance(hybridRetriever, Retriever)


def testHybridRetrieverCombinesSparseAndDenseRankings() -> None:
    hybridRetriever, sparseRetriever, denseRetriever = buildHybridRetriever()

    results = hybridRetriever.retrieve("  errore durante il checkout  ", k=3)

    assert sparseRetriever.calls == [("errore durante il checkout", 2)]
    assert denseRetriever.calls == [("errore durante il checkout", 3)]
    assert [result.chunk.id for result in results] == ["checkout", "payment", "cart"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert {result.retriever for result in results} == {"rrf"}

    checkoutContributions = {contribution.retriever for contribution in results[0].contributions}
    assert checkoutContributions == {"sparse", "dense"}


def testHybridRetrieverLimitsTheFinalResultCount() -> None:
    hybridRetriever, _, _ = buildHybridRetriever()

    results = hybridRetriever.retrieve("errore durante il checkout", k=2)

    assert [result.chunk.id for result in results] == ["checkout", "payment"]


@pytest.mark.parametrize(
    ("sparseTopK", "denseTopK", "invalidParameter"),
    [
        pytest.param(0, 3, "sparse_top_k", id="sparse-top-k-non-valido"),
        pytest.param(2, 0, "dense_top_k", id="dense-top-k-non-valido"),
    ],
)
def testHybridRetrieverRejectsInvalidCandidateCounts(
    sparseTopK: int,
    denseTopK: int,
    invalidParameter: str,
) -> None:
    emptyRetriever = RecordingRetriever([])

    with pytest.raises(ValueError, match=invalidParameter):
        HybridRetriever(
            sparseRetriever=emptyRetriever,
            denseRetriever=emptyRetriever,
            rankFusion=ReciprocalRankFusion(),
            sparseTopK=sparseTopK,
            denseTopK=denseTopK,
        )


@pytest.mark.parametrize(
    ("query", "k", "expectedMessage"),
    [
        pytest.param("   ", 3, "query", id="query-vuota"),
        pytest.param("checkout failure", 0, "parametro k", id="top-k-finale-non-valido"),
    ],
)
def testHybridRetrieverRejectsInvalidRequests(
    query: str,
    k: int,
    expectedMessage: str,
) -> None:
    hybridRetriever, sparseRetriever, denseRetriever = buildHybridRetriever()

    with pytest.raises(ValueError, match=expectedMessage):
        hybridRetriever.retrieve(query, k)

    assert sparseRetriever.calls == []
    assert denseRetriever.calls == []
