import subprocess
from argparse import ArgumentParser
from pathlib import Path

from app.bootstrap import RAGApplicationFactory
from app.cli.common import (
    addApplicationConfigurationArguments,
    loadApplicationConfiguration,
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


def buildArgumentParser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Esegue una configurazione sperimentale sul golden dataset.",
    )
    parser.add_argument(
        "--experiment-config",
        dest="experimentConfig",
        type=Path,
        required=True,
        help="Percorso della configurazione sperimentale da eseguire.",
    )
    addApplicationConfigurationArguments(parser)
    return parser


def main() -> None:
    arguments = buildArgumentParser().parse_args()
    applicationConfiguration = loadApplicationConfiguration(arguments)
    experimentConfiguration = ExperimentConfigurationLoader().load(arguments.experimentConfig)

    ragService = RAGApplicationFactory(applicationConfiguration).createRagService(
        retrievalMode=experimentConfiguration.retrieval.mode,
        rerankerEnabled=experimentConfiguration.reranker.enabled,
    )
    goldenCases = GoldenCaseLoader().load(applicationConfiguration.evaluation.datasetPath)
    metadata = buildExperimentMetadata(
        applicationConfiguration,
        experimentConfiguration,
    )
    experimentRun = ExperimentRunner(
        system=ragService,
        topK=applicationConfiguration.retrieval.topK,
    ).execute(experimentConfiguration, goldenCases, metadata)
    outputDirectory = ExperimentResultWriter().save(
        experimentRun,
        applicationConfiguration.evaluation.resultsPath,
    )

    print(f"Esperimento completato: {experimentRun.runId}")
    print(f"Recall@K medio: {experimentRun.metrics.meanRecallAtK:.4f}")
    print(f"Mean Reciprocal Rank: {experimentRun.metrics.meanReciprocalRank:.4f}")
    print(f"Risultati salvati in: {outputDirectory}")


def buildExperimentMetadata(
    applicationConfiguration: ApplicationConfiguration,
    experimentConfiguration: ExperimentConfiguration,
) -> ExperimentMetadata:
    modelIdentifiers = {
        "generator": applicationConfiguration.llm.model,
    }
    retrievalMode = experimentConfiguration.retrieval.mode
    if retrievalMode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
        modelIdentifiers["embedding"] = applicationConfiguration.embeddings.model
    if experimentConfiguration.reranker.enabled:
        modelIdentifiers["reranker"] = applicationConfiguration.reranker.model

    return ExperimentMetadata(
        datasetPath=applicationConfiguration.evaluation.datasetPath.as_posix(),
        knowledgeBaseVersion=(applicationConfiguration.ingestion.knowledgeBaseVersion),
        gitCommit=readGitCommit(),
        modelIdentifiers=modelIdentifiers,
    )


def readGitCommit() -> str | None:
    try:
        completedProcess = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completedProcess.returncode != 0:
        return None

    commit = completedProcess.stdout.strip()
    return commit or None


if __name__ == "__main__":
    main()
