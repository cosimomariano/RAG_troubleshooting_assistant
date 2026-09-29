from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import Field

from app.evaluation.cases import GoldenCase
from app.evaluation.configuration import ExperimentConfiguration
from app.evaluation.metrics import RetrievalMetricsCalculator
from app.models import OperationalMetrics, SourceReference, StrictModel
from app.services import TroubleshootingSystem

TimestampProvider = Callable[[], datetime]


class ExperimentMetadata(StrictModel):
    datasetPath: str = Field(
        min_length=1,
        description="Percorso del golden dataset utilizzato",
    )
    knowledgeBaseVersion: str = Field(
        min_length=1,
        description="Versione della Knowledge Base utilizzata",
    )
    gitCommit: str | None = Field(
        default=None,
        description="Commit Git dal quale è stato eseguito l'esperimento",
    )
    modelIdentifiers: dict[str, str] = Field(
        default_factory=dict,
        description="Identificativi dei modelli impiegati dalla configurazione",
    )


class ExperimentCaseResult(StrictModel):
    caseId: str = Field(min_length=1, description="Identificativo del golden case")
    question: str = Field(min_length=1, description="Domanda sottoposta al sistema")
    expectedAnswer: str = Field(
        min_length=1,
        description="Risposta di riferimento annotata nel golden dataset",
    )
    generatedAnswer: str = Field(
        min_length=1,
        description="Risposta generata dal sistema RAG",
    )
    relevantDocuments: tuple[str, ...] = Field(
        min_length=1,
        description="Documenti rilevanti annotati per il caso",
    )
    retrievedSources: tuple[SourceReference, ...] = Field(
        description="Fonti recuperate dal sistema nell'ordine di ranking",
    )
    recallAtK: float = Field(
        ge=0,
        le=1,
        description="Recall calcolato sulle prime K fonti recuperate",
    )
    reciprocalRank: float = Field(
        ge=0,
        le=1,
        description="Inverso della posizione della prima fonte rilevante",
    )
    latencyMs: float = Field(
        ge=0,
        description="Latenza complessiva del caso in millisecondi",
    )
    operationalMetrics: OperationalMetrics = Field(
        description="Metriche operative raccolte durante il caso",
    )


class ExperimentMetrics(StrictModel):
    caseCount: int = Field(ge=1, description="Numero di golden case eseguiti")
    topK: int = Field(ge=1, description="Profondità della graduatoria valutata")
    meanRecallAtK: float = Field(
        ge=0,
        le=1,
        description="Recall@K medio sui casi eseguiti",
    )
    meanReciprocalRank: float = Field(
        ge=0,
        le=1,
        description="Mean Reciprocal Rank dei casi eseguiti",
    )
    meanLatencyMs: float = Field(
        ge=0,
        description="Latenza media complessiva in millisecondi",
    )
    meanRetrievalLatencyMs: float = Field(
        ge=0,
        description="Latenza media del retrieval in millisecondi",
    )
    meanRerankingLatencyMs: float = Field(
        ge=0,
        description="Latenza media del reranking in millisecondi",
    )
    meanPromptBuildLatencyMs: float = Field(
        ge=0,
        description="Latenza media della costruzione del prompt in millisecondi",
    )
    meanGenerationLatencyMs: float = Field(
        ge=0,
        description="Latenza media della generazione in millisecondi",
    )
    tokenUsageCaseCount: int = Field(
        ge=0,
        description="Numero di casi per i quali il provider ha comunicato i token",
    )
    totalInputTokens: int = Field(ge=0, description="Token di input complessivi")
    totalOutputTokens: int = Field(ge=0, description="Token di output complessivi")
    totalTokens: int = Field(ge=0, description="Token complessivi dell'esperimento")


class ExperimentRun(StrictModel):
    runId: str = Field(min_length=1, description="Identificativo univoco dell'esecuzione")
    executedAt: datetime = Field(description="Istante di avvio dell'esecuzione")
    configuration: ExperimentConfiguration
    metadata: ExperimentMetadata
    metrics: ExperimentMetrics
    cases: tuple[ExperimentCaseResult, ...] = Field(
        min_length=1,
        description="Risultati dei singoli golden case",
    )


class ExperimentRunner:
    """Esegue i golden case applicando la stessa procedura a ogni sistema RAG."""

    def __init__(
        self,
        system: TroubleshootingSystem,
        topK: int,
        timestampProvider: TimestampProvider | None = None,
    ) -> None:
        if topK < 1:
            raise ValueError("Il valore Top-K dell'esperimento deve essere maggiore di zero.")

        self.system = system
        self.topK = topK
        self.timestampProvider = timestampProvider or self.currentUtcTime

    def execute(
        self,
        configuration: ExperimentConfiguration,
        cases: Sequence[GoldenCase],
        metadata: ExperimentMetadata,
    ) -> ExperimentRun:
        if not cases:
            raise ValueError("È richiesto almeno un golden case per l'esperimento.")

        executionTime = self.timestampProvider()
        if executionTime.tzinfo is None or executionTime.utcoffset() is None:
            raise ValueError("Il timestamp dell'esperimento deve includere il fuso orario.")

        caseResults = tuple(self.executeCase(case) for case in cases)

        return ExperimentRun(
            runId=self.buildRunId(configuration.experiment.id, executionTime),
            executedAt=executionTime,
            configuration=configuration,
            metadata=metadata,
            metrics=self.calculateMetrics(caseResults),
            cases=caseResults,
        )

    def executeCase(self, case: GoldenCase) -> ExperimentCaseResult:
        response = self.system.troubleshoot(
            question=case.question,
            incidentContext=case.incidentContext,
        )
        retrievedSources = tuple(response.sources[: self.topK])
        retrievedDocuments = [source.source for source in retrievedSources]

        return ExperimentCaseResult(
            caseId=case.id,
            question=case.question,
            expectedAnswer=case.expectedAnswer,
            generatedAnswer=response.answer,
            relevantDocuments=case.relevantDocuments,
            retrievedSources=retrievedSources,
            recallAtK=RetrievalMetricsCalculator.recallAtK(
                retrievedDocuments,
                case.relevantDocuments,
                self.topK,
            ),
            reciprocalRank=RetrievalMetricsCalculator.reciprocalRank(
                retrievedDocuments,
                case.relevantDocuments,
            ),
            latencyMs=response.latencyMs,
            operationalMetrics=response.operationalMetrics,
        )

    def calculateMetrics(
        self,
        caseResults: Sequence[ExperimentCaseResult],
    ) -> ExperimentMetrics:
        caseCount = len(caseResults)
        meanRecall = sum(result.recallAtK for result in caseResults) / caseCount
        meanReciprocalRank = sum(result.reciprocalRank for result in caseResults) / caseCount
        meanLatencyMs = sum(result.latencyMs for result in caseResults) / caseCount
        meanRetrievalLatencyMs = (
            sum(result.operationalMetrics.retrievalLatencyMs for result in caseResults) / caseCount
        )
        meanRerankingLatencyMs = (
            sum(result.operationalMetrics.rerankingLatencyMs for result in caseResults) / caseCount
        )
        meanPromptBuildLatencyMs = (
            sum(result.operationalMetrics.promptBuildLatencyMs for result in caseResults)
            / caseCount
        )
        meanGenerationLatencyMs = (
            sum(result.operationalMetrics.generationLatencyMs for result in caseResults) / caseCount
        )
        tokenUsages = [
            result.operationalMetrics.tokenUsage
            for result in caseResults
            if result.operationalMetrics.tokenUsage is not None
        ]

        return ExperimentMetrics(
            caseCount=caseCount,
            topK=self.topK,
            meanRecallAtK=meanRecall,
            meanReciprocalRank=meanReciprocalRank,
            meanLatencyMs=meanLatencyMs,
            meanRetrievalLatencyMs=meanRetrievalLatencyMs,
            meanRerankingLatencyMs=meanRerankingLatencyMs,
            meanPromptBuildLatencyMs=meanPromptBuildLatencyMs,
            meanGenerationLatencyMs=meanGenerationLatencyMs,
            tokenUsageCaseCount=len(tokenUsages),
            totalInputTokens=sum(usage.inputTokens for usage in tokenUsages),
            totalOutputTokens=sum(usage.outputTokens for usage in tokenUsages),
            totalTokens=sum(usage.totalTokens for usage in tokenUsages),
        )

    @staticmethod
    def buildRunId(experimentId: str, executionTime: datetime) -> str:
        timestamp = executionTime.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}_{experimentId}"

    @staticmethod
    def currentUtcTime() -> datetime:
        return datetime.now(UTC)
