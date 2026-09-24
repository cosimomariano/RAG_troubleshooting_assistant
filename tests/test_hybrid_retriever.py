import pytest
from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.retrieval import HybridRetriever, ReciprocalRankFusion, Retriever

class RecordingRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self._results[:k]

def build_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=f"Contenuto diagnostico del chunk {chunk_id}.",
        metadata=SourceMetadata(
            source=f"knowledge_base/{chunk_id}.md",
            document_type="runbook",
        ),
    )

def build_result(
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

def build_hybrid_retriever() -> tuple[
    HybridRetriever,
    RecordingRetriever,
    RecordingRetriever,
]:
    payment = build_chunk("payment")
    checkout = build_chunk("checkout")
    cart = build_chunk("cart")

    sparse_retriever = RecordingRetriever(
        [
            build_result(payment, rank=1, score=8.5, retriever="sparse"),
            build_result(checkout, rank=2, score=4.2, retriever="sparse"),
        ]
    )
    dense_retriever = RecordingRetriever(
        [
            build_result(checkout, rank=1, score=0.92, retriever="dense"),
            build_result(cart, rank=2, score=0.81, retriever="dense"),
            build_result(payment, rank=3, score=0.73, retriever="dense"),
        ]
    )
    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        rank_fusion=ReciprocalRankFusion(),
        sparse_top_k=2,
        dense_top_k=3,
    )
    return hybrid_retriever, sparse_retriever, dense_retriever

def test_hybrid_retriever_satisfies_common_retriever_contract() -> None:
    hybrid_retriever, _, _ = build_hybrid_retriever()

    assert isinstance(hybrid_retriever, Retriever)

def test_hybrid_retriever_combines_sparse_and_dense_rankings() -> None:
    hybrid_retriever, sparse_retriever, dense_retriever = build_hybrid_retriever()

    results = hybrid_retriever.retrieve("  errore durante il checkout  ", k=3)

    assert sparse_retriever.calls == [("errore durante il checkout", 2)]
    assert dense_retriever.calls == [("errore durante il checkout", 3)]
    assert [result.chunk.id for result in results] == ["checkout", "payment", "cart"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert {result.retriever for result in results} == {"rrf"}

    checkout_contributions = {
        contribution.retriever for contribution in results[0].contributions
    }
    assert checkout_contributions == {"sparse", "dense"}

def test_hybrid_retriever_limits_the_final_result_count() -> None:
    hybrid_retriever, _, _ = build_hybrid_retriever()

    results = hybrid_retriever.retrieve("errore durante il checkout", k=2)

    assert [result.chunk.id for result in results] == ["checkout", "payment"]

@pytest.mark.parametrize(
    ("sparse_top_k", "dense_top_k", "invalid_parameter"),
    [
        pytest.param(0, 3, "sparse_top_k", id="sparse-top-k-non-valido"),
        pytest.param(2, 0, "dense_top_k", id="dense-top-k-non-valido"),
    ],
)
def test_hybrid_retriever_rejects_invalid_candidate_counts(
    sparse_top_k: int,
    dense_top_k: int,
    invalid_parameter: str,
) -> None:
    empty_retriever = RecordingRetriever([])

    with pytest.raises(ValueError, match=invalid_parameter):
        HybridRetriever(
            sparse_retriever=empty_retriever,
            dense_retriever=empty_retriever,
            rank_fusion=ReciprocalRankFusion(),
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
        )

@pytest.mark.parametrize(
    ("query", "k", "expected_message"),
    [
        pytest.param("   ", 3, "query", id="query-vuota"),
        pytest.param("checkout failure", 0, "parametro k", id="top-k-finale-non-valido"),
    ],
)
def test_hybrid_retriever_rejects_invalid_requests(
    query: str,
    k: int,
    expected_message: str,
) -> None:
    hybrid_retriever, sparse_retriever, dense_retriever = build_hybrid_retriever()

    with pytest.raises(ValueError, match=expected_message):
        hybrid_retriever.retrieve(query, k)

    assert sparse_retriever.calls == []
    assert dense_retriever.calls == []