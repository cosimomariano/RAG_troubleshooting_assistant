"""Smoke test del percorso completo del primo Dense RAG."""
import json
from collections.abc import Sequence
from pathlib import Path
import httpx
from fastapi.testclient import TestClient
from app.api import create_app
from app.generation import OllamaLLMClient, PromptBuilder
from app.indexing import EmbeddingVector, FaissVectorIndex
from app.ingestion import LocalDocumentLoader, RegexSensitiveDataMasker, SectionAwareChunker
from app.retrieval import DenseRetriever
from app.services import RAGService

KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge_base"

class DeterministicEmbeddingModel:
    model_name = "embedding-deterministico-per-test"

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [self._vector_for(text) for text in texts]

    @staticmethod
    def _vector_for(text: str) -> EmbeddingVector:
        normalized_text = text.lower()

        if "connection refused" in normalized_text:
            return [1.0, 0.0, 0.0]
        if "payment" in normalized_text:
            return [0.8, 0.2, 0.0]
        if "cart" in normalized_text or "carrello" in normalized_text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

def build_persistent_index(
    index_path: Path,
    embedding_model: DeterministicEmbeddingModel,
) -> FaissVectorIndex:
    # Esegue ingestione, sanitizzazione, chunking , embedding e persistenza dell'indice.

    loader = LocalDocumentLoader(KNOWLEDGE_BASE)
    masker = RegexSensitiveDataMasker()
    chunker = SectionAwareChunker(chunk_size=500)

    chunks = []
    for document in loader.load():
        sanitized_document = document.model_copy(update={"text": masker.mask(document.text)})
        chunks.extend(chunker.chunk(sanitized_document))

    vector_index = FaissVectorIndex(dimension=3)
    chunk_vectors = embedding_model.encode([chunk.text for chunk in chunks])
    vector_index.add(chunks, chunk_vectors)
    vector_index.save(index_path)

    return FaissVectorIndex.load(index_path)

def test_dense_rag_request_reaches_mocked_ollama_with_retrieved_evidence(
    tmp_path: Path,
) -> None:
    embedding_model = DeterministicEmbeddingModel()
    vector_index = build_persistent_index(tmp_path / "dense-index", embedding_model)
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_index=vector_index,
    )

    ollama_request: dict[str, object] = {}

    def handle_ollama_request(request: httpx.Request) -> httpx.Response:
        ollama_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "response": ("Il checkout non completa il pagamento perché il servizio payment non è raggiungibile."),
                "done": True,
            },
        )

    transport = httpx.MockTransport(handle_ollama_request)
    with httpx.Client(transport=transport) as http_client:
        llm_client = OllamaLLMClient(
            base_url="http://ollama-smoke-test:11434",
            model="modello-smoke-test",
            timeout_seconds=10,
            http_client=http_client,
        )
        rag_service = RAGService(
            retriever=retriever,
            prompt_builder=PromptBuilder(),
            llm_client=llm_client,
            top_k=1,
        )

        with TestClient(create_app(rag_service)) as api_client:
            response = api_client.post(
                "/troubleshoot",
                json={
                    "question": "Perché checkout non completa il pagamento?",
                    "incident_context": ("La chiamata payment/charge restituisce connection refused."),
                    "service": "checkout",
                },
            )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["answer"] == ("Il checkout non completa il pagamento perché il servizio payment non è raggiungibile." )
    assert response_body["latency_ms"] >= 0
    assert len(response_body["sources"]) == 1
    assert response_body["sources"][0]["source"] == "runbooks/payment-unreachable.md"
    assert response_body["sources"][0]["section"] == "Diagnosi"
    assert response_body["sources"][0]["chunk_id"].startswith("chunk-")

    prompt_sent_to_ollama = ollama_request["prompt"]
    assert isinstance(prompt_sent_to_ollama, str)
    assert "[FONTE 1]" in prompt_sent_to_ollama
    assert "runbooks/payment-unreachable.md" in prompt_sent_to_ollama
    assert "connection refused" in prompt_sent_to_ollama
    assert "[MASCHERATO:IP_PRIVATO]" in prompt_sent_to_ollama
    assert "10.23.4.5" not in prompt_sent_to_ollama
    assert ollama_request["model"] == "modello-smoke-test"
    assert ollama_request["stream"] is False