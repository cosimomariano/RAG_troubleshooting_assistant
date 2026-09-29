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
from app.models import OperationalMetrics, RAGResponse, SourceReference, TokenUsage
from app.retrieval import RetrievalMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicExperimentSystem:
    def __init__(self) -> None:
        self.receivedQuestions: list[str] = []

    def troubleshoot(
        self,
        question: str,
        incidentContext: str | None = None,
    ) -> RAGResponse:
        self.receivedQuestions.append(question)

        if "pagamento" in question:
            return RAGResponse(
                answer="Il servizio Payment è raggiungibile ma la chiamata Charge fallisce.",
                sources=[
                    self.source("architecture/system-overview.md", "architecture-001"),
                    self.source("runbooks/payment-failure.md", "payment-001", rank=2),
                ],
                latencyMs=10.0,
                operationalMetrics=OperationalMetrics(
                    retrievalLatencyMs=2.0,
                    rerankingLatencyMs=1.0,
                    promptBuildLatencyMs=0.5,
                    generationLatencyMs=6.5,
                    totalLatencyMs=10.0,
                    tokenUsage=TokenUsage(
                        inputTokens=10,
                        outputTokens=5,
                        totalTokens=15,
                    ),
                ),
            )

        return RAGResponse(
            answer="Le fonti recuperate non permettono di identificare la causa.",
            sources=[self.source("services/checkout-service.md", "checkout-001")],
            latencyMs=20.0,
            operationalMetrics=OperationalMetrics(
                retrievalLatencyMs=4.0,
                rerankingLatencyMs=0.0,
                promptBuildLatencyMs=1.0,
                generationLatencyMs=14.0,
                totalLatencyMs=20.0,
            ),
        )

    @staticmethod
    def source(source: str, chunkId: str, rank: int = 1) -> SourceReference:
        return SourceReference(
            citationId=f"FONTE_{rank}",
            documentId=f"document-{chunkId}",
            source=source,
            chunkId=chunkId,
            documentType="markdown",
            rank=rank,
            retriever="dense",
            score=0.9,
        )


def buildCases() -> list[GoldenCase]:
    return [
        GoldenCase(
            id="case_payment",
            question="Perché il pagamento fallisce?",
            incidentContext="PaymentService.Charge status=ERROR",
            expectedAnswer="Verificare il servizio Payment.",
            relevantDocuments=["runbooks/payment-failure.md"],
        ),
        GoldenCase(
            id="case_cart",
            question="Perché il carrello non viene svuotato?",
            incidentContext="CartService.EmptyCart status=ERROR",
            expectedAnswer="Verificare l'operazione EmptyCart.",
            relevantDocuments=["runbooks/cart-failure.md"],
        ),
    ]


def buildMetadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        datasetPath="data/golden_dataset/cases.jsonl",
        knowledgeBaseVersion="otel-demo-3.1.0-kb-v1",
        gitCommit="commit-test",
        modelIdentifiers={
            "generator": "generatore-test",
            "embedding": "embedding-test",
        },
    )


def testConfigurationLoaderValidatesExistingExperimentYaml() -> None:
    configurationPath = PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"

    configuration = ExperimentConfigurationLoader().load(configurationPath)

    assert configuration.experiment.id == "dense"
    assert configuration.retrieval.mode is RetrievalMode.DENSE
    assert configuration.reranker.enabled is False


def testConfigurationLoaderRejectsInvalidStructure(tmp_path: Path) -> None:
    configurationPath = tmp_path / "invalid.yaml"
    configurationPath.write_text("experiment:\n  id: dense\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non rispetta lo schema"):
        ExperimentConfigurationLoader().load(configurationPath)


def testRunnerCalculatesAggregateMetricsForAllCases() -> None:
    system = DeterministicExperimentSystem()
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    executionTime = datetime(2026, 9, 26, 10, 30, tzinfo=UTC)
    runner = ExperimentRunner(
        system=system,
        topK=2,
        timestampProvider=lambda: executionTime,
    )

    experimentRun = runner.execute(configuration, buildCases(), buildMetadata())

    assert system.receivedQuestions == [
        "Perché il pagamento fallisce?",
        "Perché il carrello non viene svuotato?",
    ]
    assert experimentRun.runId == "20260926T103000000000Z_dense"
    assert experimentRun.metrics.caseCount == 2
    assert experimentRun.metrics.topK == 2
    assert experimentRun.metrics.meanRecallAtK == 0.5
    assert experimentRun.metrics.meanReciprocalRank == 0.25
    assert experimentRun.metrics.meanLatencyMs == 15.0
    assert experimentRun.metrics.meanRetrievalLatencyMs == 3.0
    assert experimentRun.metrics.meanRerankingLatencyMs == 0.5
    assert experimentRun.metrics.meanPromptBuildLatencyMs == 0.75
    assert experimentRun.metrics.meanGenerationLatencyMs == 10.25
    assert experimentRun.metrics.tokenUsageCaseCount == 1
    assert experimentRun.metrics.totalInputTokens == 10
    assert experimentRun.metrics.totalOutputTokens == 5
    assert experimentRun.metrics.totalTokens == 15
    assert experimentRun.cases[0].recallAtK == 1.0
    assert experimentRun.cases[0].reciprocalRank == 0.5
    assert experimentRun.cases[1].recallAtK == 0.0


def testResultWriterCreatesReproducibleExperimentFiles(tmp_path: Path) -> None:
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    runner = ExperimentRunner(
        system=DeterministicExperimentSystem(),
        topK=2,
        timestampProvider=lambda: datetime(2026, 9, 26, 10, 30, tzinfo=UTC),
    )
    experimentRun = runner.execute(configuration, buildCases(), buildMetadata())

    outputDirectory = ExperimentResultWriter().save(experimentRun, tmp_path)

    assert {path.name for path in outputDirectory.iterdir()} == {
        "cases.jsonl",
        "config.yaml",
        "metrics.json",
        "summary.md",
    }

    savedConfiguration = yaml.safe_load(
        (outputDirectory / "config.yaml").read_text(encoding="utf-8")
    )
    savedMetrics = json.loads((outputDirectory / "metrics.json").read_text(encoding="utf-8"))
    savedCases = [
        json.loads(line)
        for line in (outputDirectory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert savedConfiguration["retrieval"] == {"mode": "dense"}
    assert savedMetrics["metadata"]["git_commit"] == "commit-test"
    assert savedMetrics["metrics"]["mean_reciprocal_rank"] == 0.25
    assert [case["case_id"] for case in savedCases] == ["case_payment", "case_cart"]
    assert "Recall@K medio: 0.5000" in (outputDirectory / "summary.md").read_text(encoding="utf-8")


def testRunnerRejectsAnEmptyDataset() -> None:
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    runner = ExperimentRunner(DeterministicExperimentSystem(), topK=2)

    with pytest.raises(ValueError, match="almeno un golden case"):
        runner.execute(configuration, [], buildMetadata())


def testResultWriterDoesNotOverwriteAnExistingRun(tmp_path: Path) -> None:
    configuration = ExperimentConfigurationLoader().load(
        PROJECT_ROOT / "configs" / "experiments" / "dense.yaml"
    )
    runner = ExperimentRunner(
        system=DeterministicExperimentSystem(),
        topK=2,
        timestampProvider=lambda: datetime(2026, 9, 26, 10, 30, tzinfo=UTC),
    )
    experimentRun = runner.execute(configuration, buildCases(), buildMetadata())
    writer = ExperimentResultWriter()
    writer.save(experimentRun, tmp_path)

    with pytest.raises(FileExistsError):
        writer.save(experimentRun, tmp_path)
