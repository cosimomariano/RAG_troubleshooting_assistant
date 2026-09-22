import pytest

from app.generation import PromptBuilder
from app.models import DocumentChunk, RetrievalResult, SourceMetadata


def build_result(
    chunk_id: str,
    text: str,
    *,
    rank: int,
    source: str,
    service: str | None = None,
    section: str | None = None,
) -> RetrievalResult:
    chunk = DocumentChunk(
        id=chunk_id,
        document_id="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source=source,
            document_type="runbook",
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


def test_prompt_separates_incident_evidence_question_and_output_format() -> None:
    result = build_result(
        "payment-unreachable-001",
        "Il checkout riceve connection refused dal servizio payment.",
        rank=1,
        source="knowledge_base/runbooks/payment-unreachable.md",
        service="payment",
        section="Diagnosi",
    )

    prompt = PromptBuilder().build(
        question="Perché il checkout non completa il pagamento?",
        incident_context="La chiamata payment/charge termina con errore.",
        documents=[result],
    )

    assert "ISTRUZIONI DI SISTEMA" in prompt
    assert "CONTESTO DELL'INCIDENTE\nLa chiamata payment/charge termina con errore." in prompt
    assert "EVIDENZE RECUPERATE" in prompt
    assert "DOMANDA DELL'UTENTE\nPerché il checkout non completa il pagamento?" in prompt
    assert "FORMATO DELLA RISPOSTA" in prompt
    assert "Causa probabile:" in prompt
    assert "Verifiche consigliate:" in prompt


def test_retrieved_chunks_keep_their_order_and_provenance() -> None:
    first_result = build_result(
        "payment-unreachable-001",
        "Il servizio payment non è raggiungibile.",
        rank=1,
        source="knowledge_base/runbooks/payment-unreachable.md",
        service="payment",
        section="Possibili cause",
    )
    second_result = build_result(
        "checkout-dependencies-002",
        "Checkout dipende dal servizio payment per completare il pagamento.",
        rank=2,
        source="knowledge_base/architecture/checkout.md",
        service="checkout",
        section="Dipendenze",
    )

    prompt = PromptBuilder().build(
        question="Quale dipendenza sta causando il problema?",
        documents=[first_result, second_result],
    )

    first_source_position = prompt.index("[FONTE 1]")
    second_source_position = prompt.index("[FONTE 2]")

    assert first_source_position < second_source_position
    assert "chunk_id: payment-unreachable-001" in prompt
    assert "documento: knowledge_base/runbooks/payment-unreachable.md" in prompt
    assert "servizio: payment" in prompt
    assert "sezione: Possibili cause" in prompt


def test_optional_metadata_is_omitted_instead_of_rendering_none() -> None:
    result = build_result(
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


def test_missing_context_and_evidence_are_explicit() -> None:
    prompt = PromptBuilder().build(
        question="Qual è la causa dell'errore?",
        documents=[],
    )

    assert "Nessun contesto aggiuntivo fornito." in prompt
    assert "Nessuna fonte documentale è stata recuperata." in prompt
    assert "Se le informazioni non sono sufficienti, dichiaralo esplicitamente." in prompt


def test_question_and_incident_context_are_trimmed() -> None:
    prompt = PromptBuilder().build(
        question="  Perché payment non risponde?  ",
        incident_context="  connection refused  ",
        documents=[],
    )

    assert "DOMANDA DELL'UTENTE\nPerché payment non risponde?" in prompt
    assert "CONTESTO DELL'INCIDENTE\nconnection refused" in prompt


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question: str) -> None:
    with pytest.raises(ValueError, match="Domanda non valorizzata."):
        PromptBuilder().build(question=question, documents=[])
