from collections.abc import Callable, Sequence
import pytest
from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.reranking import Reranker, RerankingRetriever
from app.retrieval import Retriever

class SequenceClock:
    def __init__(self, values: Sequence[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)

class RecordingRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self._results[:k]

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
        top_k: int,
    ) -> list[RetrievalResult]:
        candidate_ids = tuple(candidate.chunk.id for candidate in candidates)
        self.calls.append((query, candidate_ids, top_k))

        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: self.SCORES_BY_CHUNK_ID[candidate.chunk.id],
            reverse=True,
        )
        return [
            candidate.model_copy(
                update={
                    "rank": rank,
                    "reranker_score": self.SCORES_BY_CHUNK_ID[candidate.chunk.id],
                }
            )
            for rank, candidate in enumerate(ordered_candidates, start=1)
        ]

def build_result(chunk_id: str, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=f"Indicazioni diagnostiche per il servizio {chunk_id}.",
        metadata=SourceMetadata(
            source=f"runbooks/{chunk_id}.md",
            document_type="runbook",
            service=chunk_id,
        ),
    )
    return RetrievalResult(
        chunk=chunk,
        rank=rank,
        retriever="rrf",
        fused_score=1.0 / (60 + rank),
    )

def build_pipeline(
    clock: Callable[[], float] | None = None,
) -> tuple[RerankingRetriever, RecordingRetriever, RecordingReranker]:
    candidate_retriever = RecordingRetriever(
        [
            build_result("checkout", rank=1),
            build_result("cart", rank=2),
            build_result("payment", rank=3),
        ]
    )
    reranker = RecordingReranker()
    if clock is None:
        pipeline = RerankingRetriever(
            candidate_retriever=candidate_retriever,
            reranker=reranker,
            candidate_top_n=3,
        )
    else:
        pipeline = RerankingRetriever(
            candidate_retriever=candidate_retriever,
            reranker=reranker,
            candidate_top_n=3,
            clock=clock,
        )
    return pipeline, candidate_retriever, reranker

def test_reranking_pipeline_satisfies_the_common_retriever_contract() -> None:
    pipeline, _, reranker = build_pipeline()

    assert isinstance(pipeline, Retriever)
    assert isinstance(reranker, Reranker)

def test_reranking_pipeline_uses_top_n_candidates_and_returns_final_top_k() -> None:
    pipeline, candidate_retriever, reranker = build_pipeline()

    results = pipeline.retrieve("  errore durante il pagamento  ", k=2)

    assert candidate_retriever.calls == [("errore durante il pagamento", 3)]
    assert reranker.calls == [("errore durante il pagamento", ("checkout", "cart", "payment"), 2)]
    assert [result.chunk.id for result in results] == ["payment", "checkout"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.reranker_score for result in results] == [0.97, 0.84]
    assert all(result.fused_score is not None for result in results)

def test_reranker_is_not_called_when_retrieval_produces_no_candidates() -> None:
    candidate_retriever = RecordingRetriever([])
    reranker = RecordingReranker()
    pipeline = RerankingRetriever(candidate_retriever, reranker, candidate_top_n=5)

    results = pipeline.retrieve("errore durante il pagamento", k=2)

    assert results == []
    assert candidate_retriever.calls == [("errore durante il pagamento", 5)]
    assert reranker.calls == []


def test_retrieval_and_reranking_latencies_are_measured_separately() -> None:
    clock = SequenceClock([10.0, 10.012, 20.0, 20.034])
    pipeline, _, _ = build_pipeline(clock)

    execution = pipeline.retrieve_with_metrics("errore durante il pagamento", k=2)

    assert [result.chunk.id for result in execution.results] == ["payment", "checkout"]
    assert execution.retrieval_latency_ms == pytest.approx(12.0)
    assert execution.reranking_latency_ms == pytest.approx(34.0)

def test_candidate_top_n_must_be_positive() -> None:
    with pytest.raises(ValueError, match="candidate_top_n"):
        RerankingRetriever(
            candidate_retriever=RecordingRetriever([]),
            reranker=RecordingReranker(),
            candidate_top_n=0,
        )

@pytest.mark.parametrize(
    ("query", "top_k", "expected_message"),
    [
        pytest.param("   ", 2, "query", id="query-vuota"),
        pytest.param("payment failure", 0, "parametro k", id="top-k-non-positivo"),
        pytest.param("payment failure", 4, "Top-K finale", id="top-k-supera-top-n"),
    ],
)
def test_invalid_requests_are_rejected_before_retrieval(
    query: str,
    top_k: int,
    expected_message: str,
) -> None:
    pipeline, candidate_retriever, reranker = build_pipeline()

    with pytest.raises(ValueError, match=expected_message):
        pipeline.retrieve(query, top_k)

    assert candidate_retriever.calls == []
    assert reranker.calls == []