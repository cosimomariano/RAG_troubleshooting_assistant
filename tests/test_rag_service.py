from collections.abc import Sequence

import pytest

from app.generation import LLMClient, PromptBuilder, StubLLMClient
from app.models import DocumentChunk, RetrievalResult, SourceMetadata
from app.services import RAGService


class RecordingRetriever:
    def __init__(self, results: Sequence[RetrievalResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self.results[:k]


class EmptyResponseLLM:
    def generate(self, prompt: str) -> str:
        return "   "


def build_result() -> RetrievalResult:
    chunk = DocumentChunk(
        id="payment-unreachable-001",
        document_id="runbook-payment",
        text="Il servizio payment non è raggiungibile dal checkout.",
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/payment-unreachable.md",
            document_type="runbook",
            service="payment",
            section="Diagnosi",
        ),
    )
    return RetrievalResult(
        chunk=chunk,
        rank=1,
        score=0.93,
        retriever="dense",
    )


def test_stub_client_satisfies_llm_contract_without_network_calls() -> None:
    client = StubLLMClient("Verificare la disponibilità del servizio payment.")

    assert isinstance(client, LLMClient)
    assert client.generate("Prompt di prova") == (
        "Verificare la disponibilità del servizio payment."
    )
    assert client.received_prompts == ["Prompt di prova"]


def test_rag_service_connects_retrieval_prompt_and_generation() -> None:
    retriever = RecordingRetriever([build_result()])
    llm_client = StubLLMClient("La causa probabile è l'indisponibilità del servizio payment.")
    service = RAGService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
        top_k=3,
    )

    response = service.troubleshoot(
        question="Perché il checkout non completa il pagamento?",
        incident_context="La chiamata payment/charge restituisce connection refused.",
    )

    assert retriever.calls == [
        (
            "Perché il checkout non completa il pagamento?\n"
            "La chiamata payment/charge restituisce connection refused.",
            3,
        )
    ]
    assert len(llm_client.received_prompts) == 1
    assert "[FONTE 1]" in llm_client.received_prompts[0]
    assert response.answer == ("La causa probabile è l'indisponibilità del servizio payment.")
    assert response.latency_ms >= 0
    assert response.sources[0].model_dump() == {
        "source": "knowledge_base/runbooks/payment-unreachable.md",
        "chunk_id": "payment-unreachable-001",
        "section": "Diagnosi",
        "service": "payment",
    }


def test_question_is_used_alone_when_incident_context_is_missing() -> None:
    retriever = RecordingRetriever([])
    llm_client = StubLLMClient("Le evidenze disponibili non sono sufficienti.")
    service = RAGService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
        top_k=5,
    )

    response = service.troubleshoot("Qual è la causa dell'errore?")

    assert retriever.calls == [("Qual è la causa dell'errore?", 5)]
    assert response.sources == []
    assert "Nessuna fonte documentale è stata recuperata." in llm_client.received_prompts[0]


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(top_k: int) -> None:
    with pytest.raises(ValueError, match="Top-K deve essere maggiore di zero"):
        RAGService(
            retriever=RecordingRetriever([]),
            prompt_builder=PromptBuilder(),
            llm_client=StubLLMClient("Risposta di prova"),
            top_k=top_k,
        )


def test_empty_llm_response_is_rejected() -> None:
    service = RAGService(
        retriever=RecordingRetriever([]),
        prompt_builder=PromptBuilder(),
        llm_client=EmptyResponseLLM(),
        top_k=3,
    )

    with pytest.raises(ValueError, match="risposta vuota"):
        service.troubleshoot("Perché il checkout non risponde?")
