from pathlib import Path

import pytest
import yaml

from app.generation import PromptBuilder, StubLLMClient
from app.models import RetrievalResult
from app.retrieval import (
    NoRetrievalRetriever,
    RetrievalMode,
    Retriever,
    RetrieverSelector,
)
from app.services import RAGService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGURATIONS = [
    ("llm_only.yaml", RetrievalMode.LLM_ONLY),
    ("dense.yaml", RetrievalMode.DENSE),
    ("sparse.yaml", RetrievalMode.SPARSE),
    ("hybrid.yaml", RetrievalMode.HYBRID),
]


class EmptyRetriever:
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return []


def load_experiment_configuration(file_name: str) -> dict:
    configuration_path = PROJECT_ROOT / "configs" / "experiments" / file_name
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))

    assert isinstance(configuration, dict)
    return configuration


@pytest.mark.parametrize(("file_name", "expected_mode"), EXPERIMENT_CONFIGURATIONS)
def test_experiment_configuration_changes_only_retrieval_mode(
    file_name: str,
    expected_mode: RetrievalMode,
) -> None:
    configuration = load_experiment_configuration(file_name)
    experiment = configuration["experiment"]

    assert set(configuration) == {"experiment", "retrieval"}
    assert set(experiment) == {"id", "description"}
    assert experiment["id"] == expected_mode.value
    assert experiment["description"].strip()
    assert configuration["retrieval"] == {"mode": expected_mode.value}


def test_hybrid_rerank_configuration_enables_the_second_stage() -> None:
    configuration = load_experiment_configuration("hybrid_rerank.yaml")

    assert configuration == {
        "experiment": {
            "id": "hybrid_rerank",
            "description": (
                "RAG ibrido con fusione RRF e riordinamento tramite Cross-Encoder"
            ),
        },
        "retrieval": {"mode": RetrievalMode.HYBRID.value},
        "reranker": {"enabled": True},
    }

def test_selector_builds_llm_only_baseline_without_retrieval() -> None:
    selector = RetrieverSelector(
        dense_retriever=EmptyRetriever(),
        sparse_retriever=EmptyRetriever(),
        hybrid_retriever=EmptyRetriever(),
    )

    retriever = selector.select(RetrievalMode.LLM_ONLY)

    assert isinstance(retriever, Retriever)
    assert isinstance(retriever, NoRetrievalRetriever)
    assert retriever.retrieve("Errore durante il checkout", k=5) == []


def test_llm_only_baseline_uses_question_without_document_sources() -> None:
    llm_client = StubLLMClient("Le informazioni disponibili non consentono una diagnosi certa.")
    service = RAGService(
        retriever=NoRetrievalRetriever(),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
        top_k=5,
    )

    response = service.troubleshoot(
        question="Perché il checkout non completa il pagamento?",
        incident_context="La chiamata payment/charge restituisce connection refused.",
    )

    assert response.sources == []
    assert "Nessuna fonte documentale è stata recuperata." in llm_client.received_prompts[0]
    assert "Perché il checkout non completa il pagamento?" in llm_client.received_prompts[0]
