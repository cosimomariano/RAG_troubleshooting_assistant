from pathlib import Path

import pytest

from app.config import ApplicationConfigurationLoader
from app.retrieval import RetrievalMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def buildEnvironment(tmp_path: Path) -> dict[str, str]:
    return {
        "APP_ENV": "test",
        "SERVER_HOST": "127.0.0.1",
        "SERVER_PORT": "8080",
        "LOG_LEVEL": "DEBUG",
        "OLLAMA_BASE_URL": "http://ollama-test:11434",
        "OLLAMA_MODEL": "generatore-test",
        "OLLAMA_TIMEOUT_SECONDS": "30",
        "KNOWLEDGE_BASE_PATH": str(tmp_path / "knowledge-base"),
        "KNOWLEDGE_BASE_VERSION": "kb-test-v1",
        "EMBEDDING_MODEL": "embedding-test",
        "VECTOR_STORE_PATH": str(tmp_path / "vector-store"),
        "RETRIEVAL_MODE": "hybrid",
        "RETRIEVAL_TOP_K": "4",
        "RERANKER_ENABLED": "false",
        "RERANKER_MODEL": "reranker-test",
        "RERANKER_BATCH_SIZE": "8",
        "RERANKER_CANDIDATE_TOP_N": "12",
        "GOLDEN_DATASET_PATH": str(tmp_path / "cases.jsonl"),
        "EXPERIMENT_RESULTS_PATH": str(tmp_path / "results"),
    }


def testLoaderResolvesEnvironmentReferencesAndValidatesTypes(
    tmp_path: Path,
) -> None:
    configuration = ApplicationConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "application.yaml",
        environment=buildEnvironment(tmp_path),
    )

    assert configuration.application.environment == "test"
    assert configuration.server.port == 8080
    assert configuration.llm.timeoutSeconds == 30
    assert configuration.retrieval.mode is RetrievalMode.HYBRID
    assert configuration.retrieval.topK == 4
    assert configuration.ingestion.knowledgeBaseVersion == "kb-test-v1"
    assert configuration.evaluation.resultsPath == tmp_path / "results"


def testProcessEnvironmentOverridesEnvFile(tmp_path: Path) -> None:
    environment = buildEnvironment(tmp_path)
    environmentFile = tmp_path / ".env"
    environmentFile.write_text(
        "\n".join(f"{name}={value}" for name, value in environment.items()),
        encoding="utf-8",
    )

    configuration = ApplicationConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "application.yaml",
        environmentFile=environmentFile,
        environment={"RETRIEVAL_MODE": "sparse"},
    )

    assert configuration.retrieval.mode is RetrievalMode.SPARSE
    assert configuration.application.environment == "test"


def testLoaderReportsAMissingRequiredVariable(tmp_path: Path) -> None:
    environment = buildEnvironment(tmp_path)
    environment.pop("OLLAMA_MODEL")

    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        ApplicationConfigurationLoader().load(
            PROJECT_ROOT / "configs" / "application.yaml",
            environment=environment,
        )
