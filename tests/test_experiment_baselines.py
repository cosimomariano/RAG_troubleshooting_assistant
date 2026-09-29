from pathlib import Path

import pytest
import yaml

from app.generation import PromptBuilder, StubLLMClient
from app.retrieval import NoRetrievalRetriever, RetrievalMode
from app.services import RAGService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGURATIONS = [
    ("llm_only.yaml", RetrievalMode.LLM_ONLY),
    ("dense.yaml", RetrievalMode.DENSE),
    ("sparse.yaml", RetrievalMode.SPARSE),
    ("hybrid.yaml", RetrievalMode.HYBRID),
]


def loadExperimentConfiguration(fileName: str) -> dict:
    configurationPath = PROJECT_ROOT / "configs" / "experiments" / fileName
    configuration = yaml.safe_load(configurationPath.read_text(encoding="utf-8"))

    assert isinstance(configuration, dict)
    return configuration


@pytest.mark.parametrize(("fileName", "expectedMode"), EXPERIMENT_CONFIGURATIONS)
def testExperimentConfigurationChangesOnlyRetrievalMode(
    fileName: str,
    expectedMode: RetrievalMode,
) -> None:
    configuration = loadExperimentConfiguration(fileName)
    experiment = configuration["experiment"]

    assert set(configuration) == {"experiment", "retrieval"}
    assert set(experiment) == {"id", "description"}
    assert experiment["id"] == expectedMode.value
    assert experiment["description"].strip()
    assert configuration["retrieval"] == {"mode": expectedMode.value}


def testHybridRerankConfigurationEnablesTheSecondStage() -> None:
    configuration = loadExperimentConfiguration("hybrid_rerank.yaml")

    assert configuration == {
        "experiment": {
            "id": "hybrid_rerank",
            "description": ("RAG ibrido con fusione RRF e riordinamento tramite Cross-Encoder"),
        },
        "retrieval": {"mode": RetrievalMode.HYBRID.value},
        "reranker": {"enabled": True},
    }


def testLlmOnlyBaselineUsesQuestionWithoutDocumentSources() -> None:
    llmClient = StubLLMClient("Le informazioni disponibili non consentono una diagnosi certa.")
    service = RAGService(
        retriever=NoRetrievalRetriever(),
        promptBuilder=PromptBuilder(),
        llmClient=llmClient,
        topK=5,
    )

    response = service.troubleshoot(
        question="Perché il checkout non completa il pagamento?",
        incidentContext="La chiamata payment/charge restituisce connection refused.",
    )

    assert response.sources == []
    assert "Nessuna fonte documentale è stata recuperata." in llmClient.receivedPrompts[0]
    assert "Perché il checkout non completa il pagamento?" in llmClient.receivedPrompts[0]
