import pytest

from app.generation import CitationFormatter, PromptBuilder
from app.models import DocumentChunk, RetrievalResult, SourceMetadata


def buildResult(
    chunkId: str,
    text: str,
    *,
    rank: int,
    source: str,
    service: str | None = None,
    section: str | None = None,
) -> RetrievalResult:
    chunk = DocumentChunk(
        id=chunkId,
        documentId="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source=source,
            documentType="runbook",
            service=service,
            section=section,
        ),
    )
    return RetrievalResult(
        chunk=chunk,
        rank=rank,
        score=0.91,
        retriever="dense",
    )


def testPromptSeparatesIncidentEvidenceQuestionAndOutputFormat() -> None:
    result = buildResult(
        "payment-unreachable-001",
        "Il checkout riceve connection refused dal servizio payment.",
        rank=1,
        source="knowledge_base/runbooks/payment-unreachable.md",
        service="payment",
        section="Diagnosi",
    )

    prompt = PromptBuilder().build(
        question="Perché il checkout non completa il pagamento?",
        incidentContext="La chiamata payment/charge termina con errore.",
        documents=[result],
    )

    assert "ISTRUZIONI DI SISTEMA" in prompt
    assert "CONTESTO DELL'INCIDENTE\nLa chiamata payment/charge termina con errore." in prompt
    assert "EVIDENZE RECUPERATE" in prompt
    assert "DOMANDA DELL'UTENTE\nPerché il checkout non completa il pagamento?" in prompt
    assert "FORMATO DELLA RISPOSTA" in prompt
    assert "Causa probabile:" in prompt
    assert "Verifiche consigliate:" in prompt


def testRetrievedChunksKeepTheirOrderAndProvenance() -> None:
    firstResult = buildResult(
        "payment-unreachable-001",
        "Il servizio payment non è raggiungibile.",
        rank=1,
        source="knowledge_base/runbooks/payment-unreachable.md",
        service="payment",
        section="Possibili cause",
    )
    secondResult = buildResult(
        "checkout-dependencies-002",
        "Checkout dipende dal servizio payment per completare il pagamento.",
        rank=2,
        source="knowledge_base/architecture/checkout.md",
        service="checkout",
        section="Dipendenze",
    )

    prompt = PromptBuilder().build(
        question="Quale dipendenza sta causando il problema?",
        documents=[firstResult, secondResult],
    )

    firstSourcePosition = prompt.index("[FONTE_1]")
    secondSourcePosition = prompt.index("[FONTE_2]")

    assert firstSourcePosition < secondSourcePosition
    assert "document_id: runbook-checkout" in prompt
    assert "chunk_id: payment-unreachable-001" in prompt
    assert "documento: knowledge_base/runbooks/payment-unreachable.md" in prompt
    assert "servizio: payment" in prompt
    assert "sezione: Possibili cause" in prompt


def testOptionalMetadataIsOmittedInsteadOfRenderingNone() -> None:
    result = buildResult(
        "generic-error-001",
        "Controllare lo stato delle dipendenze del servizio.",
        rank=1,
        source="knowledge_base/runbooks/generic-error.md",
    )

    prompt = PromptBuilder().build(
        question="Quali verifiche devo effettuare?",
        documents=[result],
    )

    assert "servizio:" not in prompt
    assert "sezione:" not in prompt
    assert "None" not in prompt


def testMissingContextAndEvidenceAreExplicit() -> None:
    prompt = PromptBuilder().build(
        question="Qual è la causa dell'errore?",
        documents=[],
    )

    assert "Nessun contesto aggiuntivo fornito." in prompt
    assert "Nessuna fonte documentale è stata recuperata." in prompt
    assert "Se le informazioni non sono sufficienti, dichiaralo esplicitamente." in prompt


def testQuestionAndIncidentContextAreTrimmed() -> None:
    prompt = PromptBuilder().build(
        question="  Perché payment non risponde?  ",
        incidentContext="  connection refused  ",
        documents=[],
    )

    assert "DOMANDA DELL'UTENTE\nPerché payment non risponde?" in prompt
    assert "CONTESTO DELL'INCIDENTE\nconnection refused" in prompt


@pytest.mark.parametrize("question", ["", "   "])
def testEmptyQuestionIsRejected(question: str) -> None:
    with pytest.raises(ValueError, match="Domanda non valorizzata."):
        PromptBuilder().build(question=question, documents=[])


@pytest.mark.parametrize("position", [0, -1])
def testCitationIdentifierRequiresAPositivePosition(position: int) -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        CitationFormatter.buildIdentifier(position)
