from collections.abc import Callable, Sequence

import pytest

from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.reranking import Reranker, RerankingRetriever
from app.retrieval import Retriever


class SequenceClock:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class RecordingRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self.results[:k]


class RecordingReranker:
    SCORES_BY_CHUNK_ID = {
        "payment": 0.97,
        "checkout": 0.84,
        "cart": 0.12,
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        topK: int,
    ) -> list[RetrievalResult]:
        candidateIds = tuple(candidate.chunk.id for candidate in candidates)
        self.calls.append((query, candidateIds, topK))

        orderedCandidates = sorted(
            candidates,
            key=lambda candidate: self.SCORES_BY_CHUNK_ID[candidate.chunk.id],
            reverse=True,
        )
        return [
            candidate.model_copy(
                update={
                    "rank": rank,
                    "rerankerScore": self.SCORES_BY_CHUNK_ID[candidate.chunk.id],
                }
            )
            for rank, candidate in enumerate(orderedCandidates, start=1)
        ]


def buildResult(chunkId: str, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        id=chunkId,
        documentId=f"document-{chunkId}",
        text=f"Indicazioni diagnostiche per il servizio {chunkId}.",
        metadata=SourceMetadata(
            source=f"runbooks/{chunkId}.md",
            documentType="runbook",
            service=chunkId,
        ),
    )
    return RetrievalResult(
        chunk=chunk,
        rank=rank,
        retriever="rrf",
        fusedScore=1.0 / (60 + rank),
    )


def buildPipeline(
    clock: Callable[[], float] | None = None,
) -> tuple[RerankingRetriever, RecordingRetriever, RecordingReranker]:
    candidateRetriever = RecordingRetriever(
        [
            buildResult("checkout", rank=1),
            buildResult("cart", rank=2),
            buildResult("payment", rank=3),
        ]
    )
    reranker = RecordingReranker()
    if clock is None:
        pipeline = RerankingRetriever(
            candidateRetriever=candidateRetriever,
            reranker=reranker,
            candidateTopN=3,
        )
    else:
        pipeline = RerankingRetriever(
            candidateRetriever=candidateRetriever,
            reranker=reranker,
            candidateTopN=3,
            clock=clock,
        )
    return pipeline, candidateRetriever, reranker


def testRerankingPipelineSatisfiesTheCommonRetrieverContract() -> None:
    pipeline, _, reranker = buildPipeline()

    assert isinstance(pipeline, Retriever)
    assert isinstance(reranker, Reranker)


def testRerankingPipelineUsesTopNCandidatesAndReturnsFinalTopK() -> None:
    pipeline, candidateRetriever, reranker = buildPipeline()

    results = pipeline.retrieve("  errore durante il pagamento  ", k=2)

    assert candidateRetriever.calls == [("errore durante il pagamento", 3)]
    assert reranker.calls == [("errore durante il pagamento", ("checkout", "cart", "payment"), 2)]
    assert [result.chunk.id for result in results] == ["payment", "checkout"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.rerankerScore for result in results] == [0.97, 0.84]
    assert all(result.fusedScore is not None for result in results)


def testRerankerIsNotCalledWhenRetrievalProducesNoCandidates() -> None:
    candidateRetriever = RecordingRetriever([])
    reranker = RecordingReranker()
    pipeline = RerankingRetriever(candidateRetriever, reranker, candidateTopN=5)

    results = pipeline.retrieve("errore durante il pagamento", k=2)

    assert results == []
    assert candidateRetriever.calls == [("errore durante il pagamento", 5)]
    assert reranker.calls == []


def testRetrievalAndRerankingLatenciesAreMeasuredSeparately() -> None:
    clock = SequenceClock([10.0, 10.012, 20.0, 20.034])
    pipeline, _, _ = buildPipeline(clock)

    execution = pipeline.retrieveWithMetrics("errore durante il pagamento", k=2)

    assert [result.chunk.id for result in execution.results] == ["payment", "checkout"]
    assert execution.retrievalLatencyMs == pytest.approx(12.0)
    assert execution.rerankingLatencyMs == pytest.approx(34.0)


def testCandidateTopNMustBePositive() -> None:
    with pytest.raises(ValueError, match="candidate_top_n"):
        RerankingRetriever(
            candidateRetriever=RecordingRetriever([]),
            reranker=RecordingReranker(),
            candidateTopN=0,
        )


@pytest.mark.parametrize(
    ("query", "topK", "expectedMessage"),
    [
        pytest.param("   ", 2, "query", id="query-vuota"),
        pytest.param("payment failure", 0, "parametro k", id="top-k-non-positivo"),
        pytest.param("payment failure", 4, "Top-K finale", id="top-k-supera-top-n"),
    ],
)
def testInvalidRequestsAreRejectedBeforeRetrieval(
    query: str,
    topK: int,
    expectedMessage: str,
) -> None:
    pipeline, candidateRetriever, reranker = buildPipeline()

    with pytest.raises(ValueError, match=expectedMessage):
        pipeline.retrieve(query, topK)

    assert candidateRetriever.calls == []
    assert reranker.calls == []
