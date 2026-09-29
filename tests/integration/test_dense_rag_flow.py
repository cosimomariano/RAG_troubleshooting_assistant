"""Smoke test del percorso completo del primo Dense RAG."""

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.api import createApp
from app.generation import OllamaLLMClient, PromptBuilder
from app.indexing import EmbeddingVector, FaissVectorIndex
from app.ingestion import LocalDocumentLoader, RegexSensitiveDataMasker, SectionAwareChunker
from app.retrieval import DenseRetriever
from app.services import RAGService

KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge_base"


# Mock
class DeterministicEmbeddingModel:
    modelName = "embedding-deterministico-per-test"

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [self.vectorFor(text) for text in texts]

    @staticmethod
    def vectorFor(text: str) -> EmbeddingVector:
        normalizedText = text.lower()

        if "connection refused" in normalizedText:
            return [1.0, 0.0, 0.0]
        if "payment" in normalizedText:
            return [0.8, 0.2, 0.0]
        if "cart" in normalizedText or "carrello" in normalizedText:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def buildPersistentIndex(
    indexPath: Path,
    embeddingModel: DeterministicEmbeddingModel,
) -> FaissVectorIndex:
    # Esegue ingestione, sanitizzazione, chunking , embedding e persistenza dell'indice.

    loader = LocalDocumentLoader(KNOWLEDGE_BASE)
    masker = RegexSensitiveDataMasker()
    chunker = SectionAwareChunker(chunkSize=500)

    chunks = []
    for document in loader.load():
        sanitizedDocument = document.model_copy(update={"text": masker.mask(document.text)})
        chunks.extend(chunker.chunk(sanitizedDocument))

    vectorIndex = FaissVectorIndex(dimension=3)
    chunkVectors = embeddingModel.encode([chunk.text for chunk in chunks])
    vectorIndex.add(chunks, chunkVectors)
    vectorIndex.save(indexPath)

    return FaissVectorIndex.load(indexPath)


def testDenseRagRequestReachesMockedOllamaWithRetrievedEvidence(
    tmp_path: Path,
) -> None:
    embeddingModel = DeterministicEmbeddingModel()
    vectorIndex = buildPersistentIndex(tmp_path / "dense-index", embeddingModel)
    retriever = DenseRetriever(
        embeddingModel=embeddingModel,
        vectorIndex=vectorIndex,
    )

    ollamaRequest: dict[str, object] = {}

    def handleOllamaRequest(request: httpx.Request) -> httpx.Response:
        ollamaRequest.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "response": (
                    "Il checkout non completa il pagamento perché il servizio "
                    "payment non è raggiungibile."
                ),
                "done": True,
                "prompt_eval_count": 84,
                "eval_count": 19,
            },
        )

    transport = httpx.MockTransport(handleOllamaRequest)
    with httpx.Client(transport=transport) as httpClient:
        llmClient = OllamaLLMClient(
            baseUrl="http://ollama-smoke-test:11434",
            model="modello-smoke-test",
            timeoutSeconds=10,
            httpClient=httpClient,
        )
        ragService = RAGService(
            retriever=retriever,
            promptBuilder=PromptBuilder(),
            llmClient=llmClient,
            topK=1,
        )

        with TestClient(createApp(ragService)) as apiClient:
            response = apiClient.post(
                "/troubleshoot",
                json={
                    "question": "Perché checkout non completa il pagamento?",
                    "incident_context": (
                        "La chiamata payment/charge restituisce connection refused."
                    ),
                    "service": "checkout",
                },
            )

    assert response.status_code == 200
    responseBody = response.json()
    assert responseBody["answer"] == (
        "Il checkout non completa il pagamento perché il servizio payment non è raggiungibile."
    )
    assert responseBody["latency_ms"] >= 0
    assert responseBody["operational_metrics"]["retrieval_latency_ms"] >= 0
    assert responseBody["operational_metrics"]["generation_latency_ms"] >= 0
    assert responseBody["operational_metrics"]["total_latency_ms"] == (responseBody["latency_ms"])
    assert responseBody["operational_metrics"]["token_usage"] == {
        "input_tokens": 84,
        "output_tokens": 19,
        "total_tokens": 103,
    }
    assert len(responseBody["sources"]) == 1
    assert responseBody["sources"][0]["source"] == "runbooks/payment-unreachable.md"
    assert responseBody["sources"][0]["section"] == "Diagnosi"
    assert responseBody["sources"][0]["chunk_id"].startswith("chunk-")
    assert responseBody["sources"][0]["citation_id"] == "FONTE_1"
    assert responseBody["sources"][0]["document_id"].startswith("document-")
    assert responseBody["sources"][0]["document_type"] == "markdown"
    assert responseBody["sources"][0]["rank"] == 1
    assert responseBody["sources"][0]["retriever"] == "dense"

    promptSentToOllama = ollamaRequest["prompt"]
    assert isinstance(promptSentToOllama, str)
    assert "[FONTE_1]" in promptSentToOllama
    assert "runbooks/payment-unreachable.md" in promptSentToOllama
    assert "connection refused" in promptSentToOllama
    assert "[MASCHERATO:IP_PRIVATO]" in promptSentToOllama
    assert "10.23.4.5" not in promptSentToOllama
    assert ollamaRequest["model"] == "modello-smoke-test"
    assert ollamaRequest["stream"] is False
