import pytest

from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.retrieval import ReciprocalRankFusion


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


def testRrfCombinesRankingsAndPreservesOriginalContributions() -> None:
    payment = buildChunk("payment")
    checkout = buildChunk("checkout")
    cart = buildChunk("cart")
    sparseRanking = [
        buildResult(payment, rank=1, score=8.5, retriever="sparse"),
        buildResult(checkout, rank=2, score=4.2, retriever="sparse"),
    ]
    denseRanking = [
        buildResult(checkout, rank=1, score=0.92, retriever="dense"),
        buildResult(cart, rank=2, score=0.81, retriever="dense"),
        buildResult(payment, rank=3, score=0.73, retriever="dense"),
    ]

    results = ReciprocalRankFusion().fuse([sparseRanking, denseRanking])

    assert [result.chunk.id for result in results] == ["checkout", "payment", "cart"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].fusedScore == pytest.approx((1 / 62) + (1 / 61))
    assert results[1].fusedScore == pytest.approx((1 / 61) + (1 / 63))
    assert results[2].fusedScore == pytest.approx(1 / 62)
    assert results[0].score is None
    assert results[0].retriever == "rrf"

    checkoutContributions = {
        contribution.retriever: (contribution.rank, contribution.score)
        for contribution in results[0].contributions
    }
    assert checkoutContributions == {
        "sparse": (2, 4.2),
        "dense": (1, 0.92),
    }


def testEqualFusedScoresKeepTheFirstSeenOrder() -> None:
    sparseResult = buildResult(
        buildChunk("sparse-first"),
        rank=1,
        score=5.0,
        retriever="sparse",
    )
    denseResult = buildResult(
        buildChunk("dense-second"),
        rank=1,
        score=0.9,
        retriever="dense",
    )

    results = ReciprocalRankFusion().fuse([[sparseResult], [denseResult]])

    assert [result.chunk.id for result in results] == ["sparse-first", "dense-second"]


def testEmptyRankingsProduceNoResults() -> None:
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([]) == []
    assert fusion.fuse([[], []]) == []


def testDuplicateChunkInTheSameRankingIsRejected() -> None:
    chunk = buildChunk("duplicated")
    firstResult = buildResult(chunk, rank=1, score=5.0, retriever="sparse")
    secondResult = buildResult(chunk, rank=2, score=4.0, retriever="sparse")

    with pytest.raises(ValueError, match="duplicato"):
        ReciprocalRankFusion().fuse([[firstResult, secondResult]])


def testSameChunkIdWithDifferentContentIsRejected() -> None:
    sparseChunk = buildChunk("shared-id")
    denseChunk = buildChunk("shared-id").model_copy(
        update={"text": "Contenuto differente restituito dal dense retriever."}
    )
    sparseResult = buildResult(sparseChunk, rank=1, score=5.0, retriever="sparse")
    denseResult = buildResult(denseChunk, rank=1, score=0.9, retriever="dense")

    with pytest.raises(ValueError, match="contenuti diversi"):
        ReciprocalRankFusion().fuse([[sparseResult], [denseResult]])


def testRankConstantMustBePositive() -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        ReciprocalRankFusion(rankConstant=0)
