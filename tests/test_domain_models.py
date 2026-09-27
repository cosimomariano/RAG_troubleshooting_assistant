import pytest
from pydantic import ValidationError

from app.models import (
    Document,
    DocumentChunk,
    OperationalMetrics,
    RAGResponse,
    RetrievalResult,
    SourceMetadata,
    TokenUsage,
)


def payment_runbook_metadata() -> SourceMetadata:
    return SourceMetadata(
        source="runbooks/payment-unreachable.md",
        document_type="runbook",
        service="payment",
        section="Possibili cause",
    )


def payment_diagnostic_chunk() -> DocumentChunk:
    return DocumentChunk(
        id="payment-runbook-0001",
        document_id="payment-runbook",
        text="Verificare che il servizio payment sia raggiungibile dal checkout.",
        metadata=payment_runbook_metadata(),
    )


def test_empty_runbook_can_be_loaded_before_content_validation() -> None:
    document = Document(
        id="payment-runbook",
        text="",
        metadata=payment_runbook_metadata(),
    )

    assert document.text == ""
    assert document.metadata.source == "runbooks/payment-unreachable.md"


def test_chunk_points_back_to_payment_runbook() -> None:
    chunk = payment_diagnostic_chunk()

    assert chunk.document_id == "payment-runbook"
    assert chunk.metadata.service == "payment"
    assert chunk.metadata.section == "Possibili cause"


def test_empty_text_cannot_be_indexed_as_a_chunk() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            id="payment-runbook-0001",
            document_id="payment-runbook",
            text="",
            metadata=payment_runbook_metadata(),
        )


def test_dense_result_records_score_rank_and_originating_retriever() -> None:
    result = RetrievalResult(
        chunk=payment_diagnostic_chunk(),
        rank=1,
        score=0.91,
        retriever="dense",
    )

    assert result.rank == 1
    assert result.score == pytest.approx(0.91)
    assert result.retriever == "dense"


def test_rank_zero_is_not_a_valid_search_position() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk=payment_diagnostic_chunk(),
            rank=0,
            retriever="dense",
        )


def test_token_total_must_match_input_and_output_counts() -> None:
    with pytest.raises(ValidationError, match="totale dei token"):
        TokenUsage(
            input_tokens=12,
            output_tokens=3,
            total_tokens=14,
        )


def test_response_total_latency_must_match_operational_metrics() -> None:
    with pytest.raises(ValidationError, match="latenza totale"):
        RAGResponse(
            answer="Risposta di prova",
            sources=[],
            latency_ms=20.0,
            operational_metrics=OperationalMetrics(
                retrieval_latency_ms=2.0,
                reranking_latency_ms=0.0,
                prompt_build_latency_ms=1.0,
                generation_latency_ms=15.0,
                total_latency_ms=19.0,
            ),
        )
