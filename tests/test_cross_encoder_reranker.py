from collections.abc import Sequence

import pytest

from app.models import (
    DocumentChunk,
    RetrievalContribution,
    RetrievalResult,
    SourceMetadata,
)
from app.reranking import CrossEncoderReranker, Reranker


class RecordingCrossEncoder:
    def __init__(self, scores: Sequence[object]) -> None:
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Sequence[object]:
        self.calls.append(
            {
                "inputs": inputs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return self.scores


def buildResult(chunkId: str, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        id=chunkId,
        documentId=f"document-{chunkId}",
        text=f"Procedura diagnostica relativa al servizio {chunkId}.",
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
        contributions=(RetrievalContribution(retriever="dense", rank=rank, score=0.9),),
    )


def testCrossEncoderAdapterRespectsRerankerContract() -> None:
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        backend=RecordingCrossEncoder([]),
    )

    assert isinstance(reranker, Reranker)
    assert reranker.modelName == "test-cross-encoder"
    assert reranker.batchSize == 32


def testCrossEncoderScoresQueryChunkPairsInOneBatch() -> None:
    backend = RecordingCrossEncoder([0.31, 0.96, 0.57])
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        batchSize=8,
        backend=backend,
    )
    candidates = [
        buildResult("checkout", rank=1),
        buildResult("payment", rank=2),
        buildResult("cart", rank=3),
    ]

    results = reranker.rerank("  payment non raggiungibile  ", candidates, topK=2)

    assert backend.calls == [
        {
            "inputs": [
                (
                    "payment non raggiungibile",
                    "Procedura diagnostica relativa al servizio checkout.",
                ),
                (
                    "payment non raggiungibile",
                    "Procedura diagnostica relativa al servizio payment.",
                ),
                (
                    "payment non raggiungibile",
                    "Procedura diagnostica relativa al servizio cart.",
                ),
            ],
            "batch_size": 8,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
    ]
    assert [result.chunk.id for result in results] == ["payment", "cart"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.rerankerScore for result in results] == [0.96, 0.57]
    assert results[0].fusedScore == candidates[1].fusedScore
    assert results[0].contributions == candidates[1].contributions


def testEqualScoresPreserveTheCandidateOrder() -> None:
    backend = RecordingCrossEncoder([0.8, 0.8])
    reranker = CrossEncoderReranker("test-cross-encoder", backend=backend)
    candidates = [
        buildResult("checkout", rank=1),
        buildResult("payment", rank=2),
    ]

    results = reranker.rerank("errore durante il checkout", candidates, topK=2)

    assert [result.chunk.id for result in results] == ["checkout", "payment"]


def testEmptyCandidateListSkipsModelInference() -> None:
    backend = RecordingCrossEncoder([])
    reranker = CrossEncoderReranker("test-cross-encoder", backend=backend)

    assert reranker.rerank("errore durante il checkout", [], topK=3) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("modelName", "batch_size"),
    [
        pytest.param("", 32, id="nome-modello-vuoto"),
        pytest.param("   ", 32, id="nome-modello-con-soli-spazi"),
        pytest.param("test-cross-encoder", 0, id="batch-size-zero"),
        pytest.param("test-cross-encoder", -1, id="batch-size-negativo"),
    ],
)
def testInvalidCrossEncoderConfigurationIsRejected(
    modelName: str,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(
            modelName,
            batchSize=batch_size,
            backend=RecordingCrossEncoder([]),
        )


@pytest.mark.parametrize(
    ("query", "topK", "expectedMessage"),
    [
        pytest.param("   ", 2, "query", id="query-vuota"),
        pytest.param("payment failure", 0, "Top-K", id="top-k-zero"),
    ],
)
def testInvalidRerankingRequestIsRejected(
    query: str,
    topK: int,
    expectedMessage: str,
) -> None:
    backend = RecordingCrossEncoder([0.5])
    reranker = CrossEncoderReranker("test-cross-encoder", backend=backend)

    with pytest.raises(ValueError, match=expectedMessage):
        reranker.rerank(query, [buildResult("payment", rank=1)], topK)

    assert backend.calls == []


def testBackendMustReturnOneScoreForEachCandidate() -> None:
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        backend=RecordingCrossEncoder([0.5]),
    )
    candidates = [
        buildResult("checkout", rank=1),
        buildResult("payment", rank=2),
    ]

    with pytest.raises(ValueError, match="non coincide"):
        reranker.rerank("payment failure", candidates, topK=2)


@pytest.mark.parametrize(
    "invalidScore",
    [
        pytest.param("non-numerico", id="punteggio-non-numerico"),
        pytest.param(float("nan"), id="punteggio-nan"),
        pytest.param(float("inf"), id="punteggio-infinito"),
    ],
)
def testBackendScoresMustBeFiniteNumbers(invalidScore: object) -> None:
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        backend=RecordingCrossEncoder([invalidScore]),
    )

    with pytest.raises(ValueError, match="punteggio"):
        reranker.rerank(
            "payment failure",
            [buildResult("payment", rank=1)],
            topK=1,
        )
