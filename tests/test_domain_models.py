import pytest
from pydantic import ValidationError

from app.models import Document, DocumentChunk, RetrievalResult, SourceMetadata


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
