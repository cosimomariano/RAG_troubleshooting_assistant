import json
from pathlib import Path

import yaml

from app.evaluation.experiment import ExperimentRun


class ExperimentResultWriter:
    """Salva configurazione, risultati e riepilogo di un esperimento."""

    def save(self, experimentRun: ExperimentRun, resultsRoot: str | Path) -> Path:
        outputDirectory = Path(resultsRoot) / experimentRun.runId
        outputDirectory.mkdir(parents=True, exist_ok=False)

        self.writeConfiguration(experimentRun, outputDirectory)
        self.writeMetrics(experimentRun, outputDirectory)
        self.writeCases(experimentRun, outputDirectory)
        self.writeSummary(experimentRun, outputDirectory)
        return outputDirectory

    @staticmethod
    def writeConfiguration(experimentRun: ExperimentRun, outputDirectory: Path) -> None:
        configuration = experimentRun.configuration.model_dump(mode="json")
        serializedConfiguration = yaml.safe_dump(
            configuration,
            allow_unicode=True,
            sort_keys=False,
        )
        (outputDirectory / "config.yaml").write_text(
            serializedConfiguration,
            encoding="utf-8",
        )

    @staticmethod
    def writeMetrics(experimentRun: ExperimentRun, outputDirectory: Path) -> None:
        metricsDocument = {
            "run_id": experimentRun.runId,
            "experiment_id": experimentRun.configuration.experiment.id,
            "executed_at": experimentRun.executedAt.isoformat(),
            "metadata": experimentRun.metadata.model_dump(mode="json"),
            "metrics": experimentRun.metrics.model_dump(mode="json"),
        }
        (outputDirectory / "metrics.json").write_text(
            json.dumps(metricsDocument, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def writeCases(experimentRun: ExperimentRun, outputDirectory: Path) -> None:
        serializedCases = "\n".join(
            caseResult.model_dump_json() for caseResult in experimentRun.cases
        )
        (outputDirectory / "cases.jsonl").write_text(
            serializedCases + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def writeSummary(experimentRun: ExperimentRun, outputDirectory: Path) -> None:
        metrics = experimentRun.metrics
        summary = (
            f"# Risultati esperimento {experimentRun.configuration.experiment.id}\n\n"
            f"- Esecuzione: `{experimentRun.runId}`\n"
            f"- Casi valutati: {metrics.caseCount}\n"
            f"- Top-K: {metrics.topK}\n"
            f"- Recall@K medio: {metrics.meanRecallAtK:.4f}\n"
            f"- Mean Reciprocal Rank: {metrics.meanReciprocalRank:.4f}\n"
            f"- Latenza media totale: {metrics.meanLatencyMs:.2f} ms\n"
            f"- Latenza media retrieval: {metrics.meanRetrievalLatencyMs:.2f} ms\n"
            f"- Latenza media reranking: {metrics.meanRerankingLatencyMs:.2f} ms\n"
            f"- Latenza media generazione: {metrics.meanGenerationLatencyMs:.2f} ms\n"
            f"- Token complessivi: {metrics.totalTokens}\n"
        )
        (outputDirectory / "summary.md").write_text(summary, encoding="utf-8")
