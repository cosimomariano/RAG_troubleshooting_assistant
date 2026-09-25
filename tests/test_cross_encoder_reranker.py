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
        self._scores = scores
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
        return self._scores

def build_result(chunk_id: str, rank: int) -> RetrievalResult:
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=f"Procedura diagnostica relativa al servizio {chunk_id}.",
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
        contributions=(RetrievalContribution(retriever="dense", rank=rank, score=0.9),),
    )

def test_cross_encoder_adapter_respects_reranker_contract() -> None:
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        backend=RecordingCrossEncoder([]),
    )

    assert isinstance(reranker, Reranker)
    assert reranker.model_name == "test-cross-encoder"
    assert reranker.batch_size == 32

def test_cross_encoder_scores_query_chunk_pairs_in_one_batch() -> None:
    backend = RecordingCrossEncoder([0.31, 0.96, 0.57])
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        batch_size=8,
        backend=backend,
    )
    candidates = [
        build_result("checkout", rank=1),
        build_result("payment", rank=2),
        build_result("cart", rank=3),
    ]

    results = reranker.rerank("  payment non raggiungibile  ", candidates, top_k=2)

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
    assert [result.reranker_score for result in results] == [0.96, 0.57]
    assert results[0].fused_score == candidates[1].fused_score
    assert results[0].contributions == candidates[1].contributions

def test_equal_scores_preserve_the_candidate_order() -> None:
    backend = RecordingCrossEncoder([0.8, 0.8])
    reranker = CrossEncoderReranker("test-cross-encoder", backend=backend)
    candidates = [
        build_result("checkout", rank=1),
        build_result("payment", rank=2),
    ]

    results = reranker.rerank("errore durante il checkout", candidates, top_k=2)

    assert [result.chunk.id for result in results] == ["checkout", "payment"]

def test_empty_candidate_list_skips_model_inference() -> None:
    backend = RecordingCrossEncoder([])
    reranker = CrossEncoderReranker("test-cross-encoder", backend=backend)

    assert reranker.rerank("errore durante il checkout", [], top_k=3) == []
    assert backend.calls == []

@pytest.mark.parametrize(
    ("model_name", "batch_size"),
    [
        pytest.param("", 32, id="nome-modello-vuoto"),
        pytest.param("   ", 32, id="nome-modello-con-soli-spazi"),
        pytest.param("test-cross-encoder", 0, id="batch-size-zero"),
        pytest.param("test-cross-encoder", -1, id="batch-size-negativo"),
    ],
)
def test_invalid_cross_encoder_configuration_is_rejected(
    model_name: str,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(
            model_name,
            batch_size=batch_size,
            backend=RecordingCrossEncoder([]),
        )

@pytest.mark.parametrize(
    ("query", "top_k", "expected_message"),
    [
        pytest.param("   ", 2, "query", id="query-vuota"),
        pytest.param("payment failure", 0, "Top-K", id="top-k-zero"),
    ],
)
def test_invalid_reranking_request_is_rejected(
    query: str,
    top_k: int,
    expected_message: str,
) -> None:
    backend = RecordingCrossEncoder([0.5])
    reranker = CrossEncoderReranker("test-cross-encoder", backend=backend)

    with pytest.raises(ValueError, match=expected_message):
        reranker.rerank(query, [build_result("payment", rank=1)], top_k)

    assert backend.calls == []

def test_backend_must_return_one_score_for_each_candidate() -> None:
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        backend=RecordingCrossEncoder([0.5]),
    )
    candidates = [
        build_result("checkout", rank=1),
        build_result("payment", rank=2),
    ]

    with pytest.raises(ValueError, match="non coincide"):
        reranker.rerank("payment failure", candidates, top_k=2)

@pytest.mark.parametrize(
    "invalid_score",
    [
        pytest.param("non-numerico", id="punteggio-non-numerico"),
        pytest.param(float("nan"), id="punteggio-nan"),
        pytest.param(float("inf"), id="punteggio-infinito"),
    ],
)
def test_backend_scores_must_be_finite_numbers(invalid_score: object) -> None:
    reranker = CrossEncoderReranker(
        "test-cross-encoder",
        backend=RecordingCrossEncoder([invalid_score]),
    )

    with pytest.raises(ValueError, match="punteggio"):
        reranker.rerank(
            "payment failure",
            [build_result("payment", rank=1)],
            top_k=1,
        )