from collections.abc import Sequence
from pathlib import Path

import pytest

from app.bootstrap import KnowledgeBaseIndexer, RAGApplicationFactory
from app.config import ApplicationConfiguration, ApplicationConfigurationLoader
from app.generation import StubLLMClient
from app.indexing import EmbeddingVector, FaissVectorIndex
from app.models import RetrievalResult
from app.reranking import RerankingRetriever
from app.retrieval import NoRetrievalRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = PROJECT_ROOT / "tests" / "fixtures" / "knowledge_base"


class ReadableEmbeddingModel:
    modelName = "embedding-test"

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        return [self.encodeText(text) for text in texts]

    @staticmethod
    def encodeText(text: str) -> EmbeddingVector:
        normalizedText = text.casefold()
        if "payment" in normalizedText or "pagamento" in normalizedText:
            return [1.0, 0.0, 0.0]
        if "cart" in normalizedText or "carrello" in normalizedText:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        topK: int,
    ) -> list[RetrievalResult]:
        self.calls.append((query, len(candidates), topK))
        return [
            candidate.model_copy(
                update={
                    "rank": rank,
                    "rerankerScore": float(len(candidates) - rank + 1),
                }
            )
            for rank, candidate in enumerate(candidates[:topK], start=1)
        ]


def buildConfiguration(
    tmp_path: Path,
    retrievalMode: str = "dense",
    *,
    rerankerEnabled: bool = False,
    topK: int = 1,
    candidateTopN: int = 2,
) -> ApplicationConfiguration:
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
        "RETRIEVAL_MODE": retrievalMode,
        "RETRIEVAL_TOP_K": str(topK),
        "RERANKER_ENABLED": str(rerankerEnabled).lower(),
        "RERANKER_MODEL": "reranker-test",
        "RERANKER_BATCH_SIZE": "4",
        "RERANKER_CANDIDATE_TOP_N": str(candidateTopN),
        "GOLDEN_DATASET_PATH": str(tmp_path / "cases.jsonl"),
        "EXPERIMENT_RESULTS_PATH": str(tmp_path / "results"),
    }
    return ApplicationConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "application.yaml",
        environment=environment,
    )


def testIndexerBuildsAReloadableDenseIndex(tmp_path: Path) -> None:
    configuration = buildConfiguration(tmp_path)
    report = KnowledgeBaseIndexer(
        configuration,
        embeddingModel=ReadableEmbeddingModel(),
    ).build()

    reloadedIndex = FaissVectorIndex.load(configuration.vectorStore.path)

    assert report.documentCount == 2
    assert report.chunkCount == reloadedIndex.getSize()
    assert report.vectorDimension == 3
    assert report.outputPath == configuration.vectorStore.path


def testFactoryConnectsPersistedIndexRetrievalPromptAndGeneration(
    tmp_path: Path,
) -> None:
    configuration = buildConfiguration(tmp_path)
    embeddingModel = ReadableEmbeddingModel()
    KnowledgeBaseIndexer(configuration, embeddingModel=embeddingModel).build()
    llmClient = StubLLMClient("Il servizio Payment non è raggiungibile. [FONTE_1]")

    ragService = RAGApplicationFactory(
        configuration,
        embeddingModel=embeddingModel,
        llmClient=llmClient,
    ).createRagService()
    response = ragService.troubleshoot(
        question="Perché il pagamento non viene completato?",
        incidentContext="payment/charge restituisce connection refused",
    )

    assert response.sources[0].source == "runbooks/payment-unreachable.md"
    assert "[FONTE_1]" in llmClient.receivedPrompts[0]


def testSparseFactoryDoesNotRequireAPersistedFaissIndex(tmp_path: Path) -> None:
    configuration = buildConfiguration(tmp_path, retrievalMode="sparse")
    ragService = RAGApplicationFactory(
        configuration,
        llmClient=StubLLMClient("Verificare il runbook recuperato. [FONTE_1]"),
    ).createRagService()

    response = ragService.troubleshoot("Errore payment connection refused")

    assert response.sources
    assert response.sources[0].retriever == "sparse"


def testFactoryCreatesLlmOnlyBaselineWithoutIndexes(tmp_path: Path) -> None:
    configuration = buildConfiguration(tmp_path, retrievalMode="llm_only")
    ragService = RAGApplicationFactory(
        configuration,
        llmClient=StubLLMClient("Le informazioni non sono sufficienti."),
    ).createRagService()

    response = ragService.troubleshoot("Perché il checkout non risponde?")

    assert isinstance(ragService.retriever, NoRetrievalRetriever)
    assert response.sources == []


def testHybridFactoryCombinesSparseAndDenseRetrieval(tmp_path: Path) -> None:
    configuration = buildConfiguration(tmp_path, retrievalMode="hybrid")
    embeddingModel = ReadableEmbeddingModel()
    KnowledgeBaseIndexer(configuration, embeddingModel=embeddingModel).build()
    ragService = RAGApplicationFactory(
        configuration,
        embeddingModel=embeddingModel,
        llmClient=StubLLMClient("Consultare il runbook recuperato. [FONTE_1]"),
    ).createRagService()

    response = ragService.troubleshoot("Errore payment connection refused")

    assert response.sources
    assert response.sources[0].retriever == "rrf"


def testFactoryAppliesConfiguredCandidateCountToReranking(tmp_path: Path) -> None:
    configuration = buildConfiguration(
        tmp_path,
        retrievalMode="sparse",
        rerankerEnabled=True,
        topK=1,
        candidateTopN=2,
    )
    reranker = RecordingReranker()
    ragService = RAGApplicationFactory(
        configuration,
        llmClient=StubLLMClient("Consultare la prima fonte. [FONTE_1]"),
        reranker=reranker,
    ).createRagService()

    response = ragService.troubleshoot("Errore payment connection refused")

    assert isinstance(ragService.retriever, RerankingRetriever)
    assert ragService.retriever.candidateTopN == 2
    assert reranker.calls == [("Errore payment connection refused", 2, 1)]
    assert response.sources[0].rerankerScore == 2.0


def testFactoryRejectsRerankingForLlmOnlyBaseline(tmp_path: Path) -> None:
    configuration = buildConfiguration(
        tmp_path,
        retrievalMode="llm_only",
        rerankerEnabled=True,
    )

    with pytest.raises(ValueError, match="LLM-only"):
        RAGApplicationFactory(
            configuration,
            llmClient=StubLLMClient("Risposta non utilizzata."),
        ).createRagService()
