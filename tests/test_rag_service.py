from collections.abc import Sequence

import pytest

from app.generation import LLMClient, PromptBuilder, StubLLMClient
from app.models import (
    DocumentChunk,
    LLMGenerationResult,
    RetrievalResult,
    SourceMetadata,
    TokenUsage,
)
from app.retrieval import RetrievalExecution
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

    def generateWithMetrics(self, prompt: str) -> LLMGenerationResult:
        return LLMGenerationResult.model_construct(text="", tokenUsage=None)


class SequenceClock:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class MeasuredRecordingRetriever(RecordingRetriever):
    def retrieveWithMetrics(self, query: str, k: int) -> RetrievalExecution:
        self.calls.append((query, k))
        return RetrievalExecution(
            results=tuple(self.results[:k]),
            retrievalLatencyMs=12.0,
            rerankingLatencyMs=34.0,
        )


def buildResult() -> RetrievalResult:
    chunk = DocumentChunk(
        id="payment-unreachable-001",
        documentId="runbook-payment",
        text="Il servizio payment non è raggiungibile dal checkout.",
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/payment-unreachable.md",
            documentType="runbook",
            service="payment",
            section="Diagnosi",
            category="service-unavailable",
        ),
    )
    return RetrievalResult(
        chunk=chunk,
        rank=1,
        score=None,
        retriever="rrf",
        fusedScore=0.0328,
        rerankerScore=4.82,
    )


def testStubClientSatisfiesLlmContractWithoutNetworkCalls() -> None:
    client = StubLLMClient("Verificare la disponibilità del servizio payment.")

    assert isinstance(client, LLMClient)
    assert client.generate("Prompt di prova") == (
        "Verificare la disponibilità del servizio payment."
    )
    assert client.receivedPrompts == ["Prompt di prova"]


def testRagServiceConnectsRetrievalPromptAndGeneration() -> None:
    retriever = RecordingRetriever([buildResult()])
    llmClient = StubLLMClient(
        "La causa probabile è l'indisponibilità del servizio payment.",
        tokenUsage=TokenUsage(inputTokens=12, outputTokens=3, totalTokens=15),
    )
    clock = SequenceClock([10.0, 10.1, 10.3, 10.3, 10.35, 10.35, 10.75, 10.8])
    service = RAGService(
        retriever=retriever,
        promptBuilder=PromptBuilder(),
        llmClient=llmClient,
        topK=3,
        clock=clock,
    )

    response = service.troubleshoot(
        question="Perché il checkout non completa il pagamento?",
        incidentContext="La chiamata payment/charge restituisce connection refused.",
    )

    assert retriever.calls == [
        (
            "Perché il checkout non completa il pagamento?\n"
            "La chiamata payment/charge restituisce connection refused.",
            3,
        )
    ]
    assert len(llmClient.receivedPrompts) == 1
    assert "[FONTE_1]" in llmClient.receivedPrompts[0]
    assert response.answer == ("La causa probabile è l'indisponibilità del servizio payment.")
    assert response.latencyMs == pytest.approx(800.0)
    assert response.operationalMetrics.retrievalLatencyMs == pytest.approx(200.0)
    assert response.operationalMetrics.rerankingLatencyMs == 0.0
    assert response.operationalMetrics.promptBuildLatencyMs == pytest.approx(50.0)
    assert response.operationalMetrics.generationLatencyMs == pytest.approx(400.0)
    assert response.operationalMetrics.tokenUsage == TokenUsage(
        inputTokens=12,
        outputTokens=3,
        totalTokens=15,
    )
    assert response.sources[0].model_dump() == {
        "citation_id": "FONTE_1",
        "document_id": "runbook-payment",
        "source": "knowledge_base/runbooks/payment-unreachable.md",
        "chunk_id": "payment-unreachable-001",
        "document_type": "runbook",
        "section": "Diagnosi",
        "service": "payment",
        "category": "service-unavailable",
        "rank": 1,
        "retriever": "rrf",
        "score": None,
        "fused_score": 0.0328,
        "reranker_score": 4.82,
    }


def testQuestionIsUsedAloneWhenIncidentContextIsMissing() -> None:
    retriever = RecordingRetriever([])
    llmClient = StubLLMClient("Le evidenze disponibili non sono sufficienti.")
    service = RAGService(
        retriever=retriever,
        promptBuilder=PromptBuilder(),
        llmClient=llmClient,
        topK=5,
    )

    response = service.troubleshoot("Qual è la causa dell'errore?")

    assert retriever.calls == [("Qual è la causa dell'errore?", 5)]
    assert response.sources == []
    assert "Nessuna fonte documentale è stata recuperata." in llmClient.receivedPrompts[0]


def testRerankingLatencyIsPropagatedFromAMeasuredRetriever() -> None:
    retriever = MeasuredRecordingRetriever([buildResult()])
    clock = SequenceClock([20.0, 20.1, 20.12, 20.12, 20.4, 20.45])
    service = RAGService(
        retriever=retriever,
        promptBuilder=PromptBuilder(),
        llmClient=StubLLMClient("Risposta di prova."),
        topK=1,
        clock=clock,
    )

    response = service.troubleshoot("Perché il pagamento fallisce?")

    assert response.operationalMetrics.retrievalLatencyMs == 12.0
    assert response.operationalMetrics.rerankingLatencyMs == 34.0
    assert response.operationalMetrics.promptBuildLatencyMs == pytest.approx(20.0)
    assert response.operationalMetrics.generationLatencyMs == pytest.approx(280.0)
    assert response.operationalMetrics.totalLatencyMs == pytest.approx(450.0)


@pytest.mark.parametrize("topK", [0, -1])
def testTopKMustBePositive(topK: int) -> None:
    with pytest.raises(ValueError, match="Top-K deve essere maggiore di zero"):
        RAGService(
            retriever=RecordingRetriever([]),
            promptBuilder=PromptBuilder(),
            llmClient=StubLLMClient("Risposta di prova"),
            topK=topK,
        )


def testEmptyLlmResponseIsRejected() -> None:
    service = RAGService(
        retriever=RecordingRetriever([]),
        promptBuilder=PromptBuilder(),
        llmClient=EmptyResponseLLM(),
        topK=3,
    )

    with pytest.raises(ValueError, match="risposta vuota"):
        service.troubleshoot("Perché il checkout non risponde?")
