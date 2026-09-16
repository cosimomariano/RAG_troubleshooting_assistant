import pytest
from pydantic import ValidationError
from app.models import Document, DocumentChunk, RetrievalResult, SourceMetadata

def build_metadata() -> SourceMetadata:
    return SourceMetadata( source="runbooks/payment-unreachable.md", document_type="runbook", service="payment", section="Possible causes")

def test_document_accepts_empty_source_text() -> None:
    document = Document(id="payment-runbook", text="", metadata=build_metadata())

    assert document.text == ""
    assert document.metadata.source == "runbooks/payment-unreachable.md"

def test_document_chunk_preserves_document_provenance() -> None:
    chunk = DocumentChunk( id="payment-runbook-0001", document_id="payment-runbook", text="Verificare che il servizio payment sia raggiungibile.", metadata=build_metadata())

    assert chunk.document_id == "payment-runbook"
    assert chunk.metadata.service == "payment"
    assert chunk.metadata.section == "Possible causes"

def test_document_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk( id="payment-runbook-0001", document_id="payment-runbook", text="", metadata=build_metadata())

def test_retrieval_result_contains_rank_score_and_retriever() -> None:
    chunk = DocumentChunk( id="payment-runbook-0001", document_id="payment-runbook", text="Verificare che il servizio payment sia raggiungibile.", metadata=build_metadata())
    result = RetrievalResult(chunk=chunk, rank=1, score=0.91, retriever="dense")

    assert result.rank == 1
    assert result.score == pytest.approx(0.91)
    assert result.retriever == "dense"

def test_retrieval_result_rejects_non_positive_rank() -> None:
    chunk = DocumentChunk( id="payment-runbook-0001", document_id="payment-runbook", text="Verificare che il servizio payment sia raggiungibile.", metadata=build_metadata())

    with pytest.raises(ValidationError):
        RetrievalResult(chunk=chunk, rank=0, retriever="dense")