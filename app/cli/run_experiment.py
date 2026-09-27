import subprocess
from argparse import ArgumentParser
from pathlib import Path

from app.bootstrap import RAGApplicationFactory
from app.cli.common import (
    add_application_configuration_arguments,
    load_application_configuration,
)
from app.config import ApplicationConfiguration
from app.evaluation import (
    ExperimentConfiguration,
    ExperimentConfigurationLoader,
    ExperimentMetadata,
    ExperimentResultWriter,
    ExperimentRunner,
    GoldenCaseLoader,
)
from app.retrieval import RetrievalMode


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Esegue una configurazione sperimentale sul golden dataset.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Percorso della configurazione sperimentale da eseguire.",
    )
    add_application_configuration_arguments(parser)
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    application_configuration = load_application_configuration(arguments)
    experiment_configuration = ExperimentConfigurationLoader().load(
        arguments.experiment_config
    )

    rag_service = RAGApplicationFactory(application_configuration).create_rag_service(
        retrieval_mode=experiment_configuration.retrieval.mode,
        reranker_enabled=experiment_configuration.reranker.enabled,
    )
    golden_cases = GoldenCaseLoader().load(
        application_configuration.evaluation.dataset_path
    )
    metadata = build_experiment_metadata(
        application_configuration,
        experiment_configuration,
    )
    experiment_run = ExperimentRunner(
        system=rag_service,
        top_k=application_configuration.retrieval.top_k,
    ).execute(experiment_configuration, golden_cases, metadata)
    output_directory = ExperimentResultWriter().save(
        experiment_run,
        application_configuration.evaluation.results_path,
    )

    print(f"Esperimento completato: {experiment_run.run_id}")
    print(f"Recall@K medio: {experiment_run.metrics.mean_recall_at_k:.4f}")
    print(f"Mean Reciprocal Rank: {experiment_run.metrics.mean_reciprocal_rank:.4f}")
    print(f"Risultati salvati in: {output_directory}")


def build_experiment_metadata(
    application_configuration: ApplicationConfiguration,
    experiment_configuration: ExperimentConfiguration,
) -> ExperimentMetadata:
    model_identifiers = {
        "generator": application_configuration.llm.model,
    }
    retrieval_mode = experiment_configuration.retrieval.mode
    if retrieval_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
        model_identifiers["embedding"] = application_configuration.embeddings.model
    if experiment_configuration.reranker.enabled:
        model_identifiers["reranker"] = application_configuration.reranker.model

    return ExperimentMetadata(
        dataset_path=application_configuration.evaluation.dataset_path.as_posix(),
        knowledge_base_version=(
            application_configuration.ingestion.knowledge_base_version
        ),
        git_commit=read_git_commit(),
        model_identifiers=model_identifiers,
    )


def read_git_commit() -> str | None:
    try:
        completed_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed_process.returncode != 0:
        return None

    commit = completed_process.stdout.strip()
    return commit or None


if __name__ == "__main__":
    main()
