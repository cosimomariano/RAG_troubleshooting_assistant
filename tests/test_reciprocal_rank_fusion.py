import pytest

from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.retrieval import ReciprocalRankFusion


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


def test_rrf_combines_rankings_and_preserves_original_contributions() -> None:
    payment = build_chunk("payment")
    checkout = build_chunk("checkout")
    cart = build_chunk("cart")
    sparse_ranking = [
        build_result(payment, rank=1, score=8.5, retriever="sparse"),
        build_result(checkout, rank=2, score=4.2, retriever="sparse"),
    ]
    dense_ranking = [
        build_result(checkout, rank=1, score=0.92, retriever="dense"),
        build_result(cart, rank=2, score=0.81, retriever="dense"),
        build_result(payment, rank=3, score=0.73, retriever="dense"),
    ]

    results = ReciprocalRankFusion().fuse([sparse_ranking, dense_ranking])

    assert [result.chunk.id for result in results] == ["checkout", "payment", "cart"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].fused_score == pytest.approx((1 / 62) + (1 / 61))
    assert results[1].fused_score == pytest.approx((1 / 61) + (1 / 63))
    assert results[2].fused_score == pytest.approx(1 / 62)
    assert results[0].score is None
    assert results[0].retriever == "rrf"

    checkout_contributions = {
        contribution.retriever: (contribution.rank, contribution.score)
        for contribution in results[0].contributions
    }
    assert checkout_contributions == {
        "sparse": (2, 4.2),
        "dense": (1, 0.92),
    }


def test_equal_fused_scores_keep_the_first_seen_order() -> None:
    sparse_result = build_result(
        build_chunk("sparse-first"),
        rank=1,
        score=5.0,
        retriever="sparse",
    )
    dense_result = build_result(
        build_chunk("dense-second"),
        rank=1,
        score=0.9,
        retriever="dense",
    )

    results = ReciprocalRankFusion().fuse([[sparse_result], [dense_result]])

    assert [result.chunk.id for result in results] == ["sparse-first", "dense-second"]


def test_empty_rankings_produce_no_results() -> None:
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([]) == []
    assert fusion.fuse([[], []]) == []


def test_duplicate_chunk_in_the_same_ranking_is_rejected() -> None:
    chunk = build_chunk("duplicated")
    first_result = build_result(chunk, rank=1, score=5.0, retriever="sparse")
    second_result = build_result(chunk, rank=2, score=4.0, retriever="sparse")

    with pytest.raises(ValueError, match="duplicato"):
        ReciprocalRankFusion().fuse([[first_result, second_result]])


def test_same_chunk_id_with_different_content_is_rejected() -> None:
    sparse_chunk = build_chunk("shared-id")
    dense_chunk = build_chunk("shared-id").model_copy(
        update={"text": "Contenuto differente restituito dal dense retriever."}
    )
    sparse_result = build_result(sparse_chunk, rank=1, score=5.0, retriever="sparse")
    dense_result = build_result(dense_chunk, rank=1, score=0.9, retriever="dense")

    with pytest.raises(ValueError, match="contenuti diversi"):
        ReciprocalRankFusion().fuse([[sparse_result], [dense_result]])


def test_rank_constant_must_be_positive() -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        ReciprocalRankFusion(rank_constant=0)
