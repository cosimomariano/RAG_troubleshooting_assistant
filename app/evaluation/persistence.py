import json
from pathlib import Path

import yaml

from app.evaluation.experiment import ExperimentRun


class ExperimentResultWriter:
    """Salva configurazione, risultati e riepilogo di un esperimento."""

    def save(self, experiment_run: ExperimentRun, results_root: str | Path) -> Path:
        output_directory = Path(results_root) / experiment_run.run_id
        output_directory.mkdir(parents=True, exist_ok=False)

        self._write_configuration(experiment_run, output_directory)
        self._write_metrics(experiment_run, output_directory)
        self._write_cases(experiment_run, output_directory)
        self._write_summary(experiment_run, output_directory)
        return output_directory

    @staticmethod
    def _write_configuration(experiment_run: ExperimentRun, output_directory: Path) -> None:
        configuration = experiment_run.configuration.model_dump(mode="json")
        serialized_configuration = yaml.safe_dump(
            configuration,
            allow_unicode=True,
            sort_keys=False,
        )
        (output_directory / "config.yaml").write_text(
            serialized_configuration,
            encoding="utf-8",
        )

    @staticmethod
    def _write_metrics(experiment_run: ExperimentRun, output_directory: Path) -> None:
        metrics_document = {
            "run_id": experiment_run.run_id,
            "experiment_id": experiment_run.configuration.experiment.id,
            "executed_at": experiment_run.executed_at.isoformat(),
            "metadata": experiment_run.metadata.model_dump(mode="json"),
            "metrics": experiment_run.metrics.model_dump(mode="json"),
        }
        (output_directory / "metrics.json").write_text(
            json.dumps(metrics_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_cases(experiment_run: ExperimentRun, output_directory: Path) -> None:
        serialized_cases = "\n".join(
            case_result.model_dump_json() for case_result in experiment_run.cases
        )
        (output_directory / "cases.jsonl").write_text(
            serialized_cases + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_summary(experiment_run: ExperimentRun, output_directory: Path) -> None:
        metrics = experiment_run.metrics
        summary = (
            f"# Risultati esperimento {experiment_run.configuration.experiment.id}\n\n"
            f"- Esecuzione: `{experiment_run.run_id}`\n"
            f"- Casi valutati: {metrics.case_count}\n"
            f"- Top-K: {metrics.top_k}\n"
            f"- Recall@K medio: {metrics.mean_recall_at_k:.4f}\n"
            f"- Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.4f}\n"
            f"- Latenza media totale: {metrics.mean_latency_ms:.2f} ms\n"
            f"- Latenza media retrieval: {metrics.mean_retrieval_latency_ms:.2f} ms\n"
            f"- Latenza media reranking: {metrics.mean_reranking_latency_ms:.2f} ms\n"
            f"- Latenza media generazione: {metrics.mean_generation_latency_ms:.2f} ms\n"
            f"- Token complessivi: {metrics.total_tokens}\n"
        )
        (output_directory / "summary.md").write_text(summary, encoding="utf-8")
