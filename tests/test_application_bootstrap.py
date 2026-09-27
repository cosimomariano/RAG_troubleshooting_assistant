from collections.abc import Sequence
from pathlib import Path

from app.bootstrap import KnowledgeBaseIndexer, RAGApplicationFactory
from app.config import ApplicationConfiguration, ApplicationConfigurationLoader
from app.generation import StubLLMClient
from app.indexing import EmbeddingVector, FaissVectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = PROJECT_ROOT / "tests" / "fixtures" / "knowledge_base"


class ReadableEmbeddingModel:
    model_name = "embedding-test"

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [self._encode_text(text) for text in texts]

    @staticmethod
    def _encode_text(text: str) -> EmbeddingVector:
        normalized_text = text.casefold()
        if "payment" in normalized_text or "pagamento" in normalized_text:
            return [1.0, 0.0, 0.0]
        if "cart" in normalized_text or "carrello" in normalized_text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def build_configuration(tmp_path: Path, retrieval_mode: str = "dense") -> ApplicationConfiguration:
    environment = {
        "APP_ENV": "test",
        "SERVER_HOST": "127.0.0.1",
        "SERVER_PORT": "8000",
        "LOG_LEVEL": "INFO",
        "OLLAMA_BASE_URL": "http://ollama-test:11434",
        "OLLAMA_MODEL": "generatore-test",
        "OLLAMA_TIMEOUT_SECONDS": "10",
        "KNOWLEDGE_BASE_PATH": str(KNOWLEDGE_BASE),
        "KNOWLEDGE_BASE_VERSION": "kb-test-v1",
        "EMBEDDING_MODEL": "embedding-test",
        "VECTOR_STORE_PATH": str(tmp_path / "vector-store"),
        "RETRIEVAL_MODE": retrieval_mode,
        "RETRIEVAL_TOP_K": "1",
        "RERANKER_ENABLED": "false",
        "RERANKER_MODEL": "reranker-test",
        "RERANKER_BATCH_SIZE": "4",
        "RERANKER_CANDIDATE_TOP_N": "2",
        "GOLDEN_DATASET_PATH": str(tmp_path / "cases.jsonl"),
        "EXPERIMENT_RESULTS_PATH": str(tmp_path / "results"),
    }
    return ApplicationConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "application.yaml",
        environment=environment,
    )


def test_indexer_builds_a_reloadable_dense_index(tmp_path: Path) -> None:
    configuration = build_configuration(tmp_path)
    report = KnowledgeBaseIndexer(
        configuration,
        embedding_model=ReadableEmbeddingModel(),
    ).build()

    reloaded_index = FaissVectorIndex.load(configuration.vector_store.path)

    assert report.document_count == 2
    assert report.chunk_count == reloaded_index.size
    assert report.vector_dimension == 3
    assert report.output_path == configuration.vector_store.path


def test_factory_connects_persisted_index_retrieval_prompt_and_generation(
    tmp_path: Path,
) -> None:
    configuration = build_configuration(tmp_path)
    embedding_model = ReadableEmbeddingModel()
    KnowledgeBaseIndexer(configuration, embedding_model=embedding_model).build()
    llm_client = StubLLMClient("Il servizio Payment non è raggiungibile. [FONTE_1]")

    rag_service = RAGApplicationFactory(
        configuration,
        embedding_model=embedding_model,
        llm_client=llm_client,
    ).create_rag_service()
    response = rag_service.troubleshoot(
        question="Perché il pagamento non viene completato?",
        incident_context="payment/charge restituisce connection refused",
    )

    assert response.sources[0].source == "runbooks/payment-unreachable.md"
    assert "[FONTE_1]" in llm_client.received_prompts[0]


def test_sparse_factory_does_not_require_a_persisted_faiss_index(tmp_path: Path) -> None:
    configuration = build_configuration(tmp_path, retrieval_mode="sparse")
    rag_service = RAGApplicationFactory(
        configuration,
        llm_client=StubLLMClient("Verificare il runbook recuperato. [FONTE_1]"),
    ).create_rag_service()

    response = rag_service.troubleshoot("Errore payment connection refused")

    assert response.sources
    assert response.sources[0].retriever == "sparse"
