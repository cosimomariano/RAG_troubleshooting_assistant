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


def paymentRunbookMetadata() -> SourceMetadata:
    return SourceMetadata(
        source="runbooks/payment-unreachable.md",
        documentType="runbook",
        service="payment",
        section="Possibili cause",
    )


def paymentDiagnosticChunk() -> DocumentChunk:
    return DocumentChunk(
        id="payment-runbook-0001",
        documentId="payment-runbook",
        text="Verificare che il servizio payment sia raggiungibile dal checkout.",
        metadata=paymentRunbookMetadata(),
    )


def testEmptyRunbookCanBeLoadedBeforeContentValidation() -> None:
    document = Document(
        id="payment-runbook",
        text="",
        metadata=paymentRunbookMetadata(),
    )

    assert document.text == ""
    assert document.metadata.source == "runbooks/payment-unreachable.md"


def testChunkPointsBackToPaymentRunbook() -> None:
    chunk = paymentDiagnosticChunk()

    assert chunk.documentId == "payment-runbook"
    assert chunk.metadata.service == "payment"
    assert chunk.metadata.section == "Possibili cause"


def testEmptyTextCannotBeIndexedAsAChunk() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            id="payment-runbook-0001",
            documentId="payment-runbook",
            text="",
            metadata=paymentRunbookMetadata(),
        )


def testDenseResultRecordsScoreRankAndOriginatingRetriever() -> None:
    result = RetrievalResult(
        chunk=paymentDiagnosticChunk(),
        rank=1,
        score=0.91,
        retriever="dense",
    )

    assert result.rank == 1
    assert result.score == pytest.approx(0.91)
    assert result.retriever == "dense"


def testRankZeroIsNotAValidSearchPosition() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk=paymentDiagnosticChunk(),
            rank=0,
            retriever="dense",
        )


def testTokenTotalMustMatchInputAndOutputCounts() -> None:
    with pytest.raises(ValidationError, match="totale dei token"):
        TokenUsage(
            inputTokens=12,
            outputTokens=3,
            totalTokens=14,
        )


def testResponseTotalLatencyMustMatchOperationalMetrics() -> None:
    with pytest.raises(ValidationError, match="latenza totale"):
        RAGResponse(
            answer="Risposta di prova",
            sources=[],
            latencyMs=20.0,
            operationalMetrics=OperationalMetrics(
                retrievalLatencyMs=2.0,
                rerankingLatencyMs=0.0,
                promptBuildLatencyMs=1.0,
                generationLatencyMs=15.0,
                totalLatencyMs=19.0,
            ),
        )
