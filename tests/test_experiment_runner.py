import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from app.evaluation import (
    ExperimentConfigurationLoader,
    ExperimentMetadata,
    ExperimentResultWriter,
    ExperimentRunner,
    GoldenCase,
)
from app.models import RAGResponse, SourceReference
from app.retrieval import RetrievalMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicExperimentSystem:
    def __init__(self) -> None:
        self.received_questions: list[str] = []

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse:
        self.received_questions.append(question)

        if "pagamento" in question:
            return RAGResponse(
                answer="Il servizio Payment è raggiungibile ma la chiamata Charge fallisce.",
                sources=[
                    self._source("architecture/system-overview.md", "architecture-001"),
                    self._source("runbooks/payment-failure.md", "payment-001", rank=2),
                ],
                latency_ms=10.0,
            )

        return RAGResponse(
            answer="Le fonti recuperate non permettono di identificare la causa.",
            sources=[self._source("services/checkout-service.md", "checkout-001")],
            latency_ms=20.0,
        )

    @staticmethod
    def _source(source: str, chunk_id: str, rank: int = 1) -> SourceReference:
        return SourceReference(
            citation_id=f"FONTE_{rank}",
            document_id=f"document-{chunk_id}",
            source=source,
            chunk_id=chunk_id,
            document_type="markdown",
            rank=rank,
            retriever="dense",
            score=0.9,
        )


def build_cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            id="case_payment",
            question="Perché il pagamento fallisce?",
            incident_context="PaymentService.Charge status=ERROR",
            expected_answer="Verificare il servizio Payment.",
            relevant_documents=["runbooks/payment-failure.md"],
        ),
        GoldenCase(
            id="case_cart",
            question="Perché il carrello non viene svuotato?",
            incident_context="CartService.EmptyCart status=ERROR",
            expected_answer="Verificare l'operazione EmptyCart.",
            relevant_documents=["runbooks/cart-failure.md"],
        ),
    ]


def build_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        dataset_path="data/golden_dataset/cases.jsonl",
        knowledge_base_version="otel-demo-3.1.0-kb-v1",
        git_commit="commit-test",
        model_identifiers={
            "generator": "generatore-test",
            "embedding": "embedding-test",
        },
    )


def test_configuration_loader_validates_existing_experiment_yaml() -> None:
    configuration_path = PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"

    configuration = ExperimentConfigurationLoader().load(configuration_path)

    assert configuration.experiment.id == "dense"
    assert configuration.retrieval.mode is RetrievalMode.DENSE
    assert configuration.reranker.enabled is False


def test_configuration_loader_rejects_invalid_structure(tmp_path: Path) -> None:
    configuration_path = tmp_path / "invalid.yaml"
    configuration_path.write_text("experiment:\n  id: dense\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non rispetta lo schema"):
        ExperimentConfigurationLoader().load(configuration_path)


def test_runner_calculates_aggregate_metrics_for_all_cases() -> None:
    system = DeterministicExperimentSystem()
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    execution_time = datetime(2026, 9, 26, 10, 30, tzinfo=UTC)
    runner = ExperimentRunner(
        system=system,
        top_k=2,
        timestamp_provider=lambda: execution_time,
    )

    experiment_run = runner.execute(configuration, build_cases(), build_metadata())

    assert system.received_questions == [
        "Perché il pagamento fallisce?",
        "Perché il carrello non viene svuotato?",
    ]
    assert experiment_run.run_id == "20260926T103000000000Z_dense"
    assert experiment_run.metrics.case_count == 2
    assert experiment_run.metrics.top_k == 2
    assert experiment_run.metrics.mean_recall_at_k == 0.5
    assert experiment_run.metrics.mean_reciprocal_rank == 0.25
    assert experiment_run.metrics.mean_latency_ms == 15.0
    assert experiment_run.cases[0].recall_at_k == 1.0
    assert experiment_run.cases[0].reciprocal_rank == 0.5
    assert experiment_run.cases[1].recall_at_k == 0.0


def test_result_writer_creates_reproducible_experiment_files(tmp_path: Path) -> None:
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    runner = ExperimentRunner(
        system=DeterministicExperimentSystem(),
        top_k=2,
        timestamp_provider=lambda: datetime(2026, 9, 26, 10, 30, tzinfo=UTC),
    )
    experiment_run = runner.execute(configuration, build_cases(), build_metadata())

    output_directory = ExperimentResultWriter().save(experiment_run, tmp_path)

    assert {path.name for path in output_directory.iterdir()} == {
        "cases.jsonl",
        "config.yaml",
        "metrics.json",
        "summary.md",
    }

    saved_configuration = yaml.safe_load(
        (output_directory / "config.yaml").read_text(encoding="utf-8")
    )
    saved_metrics = json.loads(
        (output_directory / "metrics.json").read_text(encoding="utf-8")
    )
    saved_cases = [
        json.loads(line)
        for line in (output_directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert saved_configuration["retrieval"] == {"mode": "dense"}
    assert saved_metrics["metadata"]["git_commit"] == "commit-test"
    assert saved_metrics["metrics"]["mean_reciprocal_rank"] == 0.25
    assert [case["case_id"] for case in saved_cases] == ["case_payment", "case_cart"]
    assert "Recall@K medio: 0.5000" in (
        output_directory / "summary.md"
    ).read_text(encoding="utf-8")


def test_runner_rejects_an_empty_dataset() -> None:
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    runner = ExperimentRunner(DeterministicExperimentSystem(), top_k=2)

    with pytest.raises(ValueError, match="almeno un golden case"):
        runner.execute(configuration, [], build_metadata())


def test_result_writer_does_not_overwrite_an_existing_run(tmp_path: Path) -> None:
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    runner = ExperimentRunner(
        system=DeterministicExperimentSystem(),
        top_k=2,
        timestamp_provider=lambda: datetime(2026, 9, 26, 10, 30, tzinfo=UTC),
    )
    experiment_run = runner.execute(configuration, build_cases(), build_metadata())
    writer = ExperimentResultWriter()
    writer.save(experiment_run, tmp_path)

    with pytest.raises(FileExistsError):
        writer.save(experiment_run, tmp_path)
