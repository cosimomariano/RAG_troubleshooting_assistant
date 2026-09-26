from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import Field

from app.evaluation.cases import GoldenCase
from app.evaluation.configuration import ExperimentConfiguration
from app.evaluation.metrics import RetrievalMetricsCalculator
from app.models import RAGResponse, SourceReference, StrictModel

TimestampProvider = Callable[[], datetime]


@runtime_checkable
class ExperimentSystem(Protocol):
    """Sistema RAG interrogabile dal runner sperimentale."""

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse: ...


class ExperimentMetadata(StrictModel):
    dataset_path: str = Field(
        min_length=1,
        description="Percorso del golden dataset utilizzato",
    )
    knowledge_base_version: str = Field(
        min_length=1,
        description="Versione della Knowledge Base utilizzata",
    )
    git_commit: str | None = Field(
        default=None,
        description="Commit Git dal quale è stato eseguito l'esperimento",
    )
    model_identifiers: dict[str, str] = Field(
        default_factory=dict,
        description="Identificativi dei modelli impiegati dalla configurazione",
    )


class ExperimentCaseResult(StrictModel):
    case_id: str = Field(min_length=1, description="Identificativo del golden case")
    question: str = Field(min_length=1, description="Domanda sottoposta al sistema")
    expected_answer: str = Field(
        min_length=1,
        description="Risposta di riferimento annotata nel golden dataset",
    )
    generated_answer: str = Field(
        min_length=1,
        description="Risposta generata dal sistema RAG",
    )
    relevant_documents: tuple[str, ...] = Field(
        min_length=1,
        description="Documenti rilevanti annotati per il caso",
    )
    retrieved_sources: tuple[SourceReference, ...] = Field(
        description="Fonti recuperate dal sistema nell'ordine di ranking",
    )
    recall_at_k: float = Field(
        ge=0,
        le=1,
        description="Recall calcolato sulle prime K fonti recuperate",
    )
    reciprocal_rank: float = Field(
        ge=0,
        le=1,
        description="Inverso della posizione della prima fonte rilevante",
    )
    latency_ms: float = Field(
        ge=0,
        description="Latenza complessiva del caso in millisecondi",
    )


class ExperimentMetrics(StrictModel):
    case_count: int = Field(ge=1, description="Numero di golden case eseguiti")
    top_k: int = Field(ge=1, description="Profondità della graduatoria valutata")
    mean_recall_at_k: float = Field(
        ge=0,
        le=1,
        description="Recall@K medio sui casi eseguiti",
    )
    mean_reciprocal_rank: float = Field(
        ge=0,
        le=1,
        description="Mean Reciprocal Rank dei casi eseguiti",
    )
    mean_latency_ms: float = Field(
        ge=0,
        description="Latenza media complessiva in millisecondi",
    )


class ExperimentRun(StrictModel):
    run_id: str = Field(min_length=1, description="Identificativo univoco dell'esecuzione")
    executed_at: datetime = Field(description="Istante di avvio dell'esecuzione")
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
        system: ExperimentSystem,
        top_k: int,
        timestamp_provider: TimestampProvider | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("Il valore Top-K dell'esperimento deve essere maggiore di zero.")

        self._system = system
        self._top_k = top_k
        self._timestamp_provider = timestamp_provider or self._current_utc_time

    def execute(
        self,
        configuration: ExperimentConfiguration,
        cases: Sequence[GoldenCase],
        metadata: ExperimentMetadata,
    ) -> ExperimentRun:
        if not cases:
            raise ValueError("È richiesto almeno un golden case per l'esperimento.")

        execution_time = self._timestamp_provider()
        if execution_time.tzinfo is None or execution_time.utcoffset() is None:
            raise ValueError("Il timestamp dell'esperimento deve includere il fuso orario.")

        case_results = tuple(self._execute_case(case) for case in cases)

        return ExperimentRun(
            run_id=self._build_run_id(configuration.experiment.id, execution_time),
            executed_at=execution_time,
            configuration=configuration,
            metadata=metadata,
            metrics=self._calculate_metrics(case_results),
            cases=case_results,
        )

    def _execute_case(self, case: GoldenCase) -> ExperimentCaseResult:
        response = self._system.troubleshoot(
            question=case.question,
            incident_context=case.incident_context,
        )
        retrieved_sources = tuple(response.sources[: self._top_k])
        retrieved_documents = [source.source for source in retrieved_sources]

        return ExperimentCaseResult(
            case_id=case.id,
            question=case.question,
            expected_answer=case.expected_answer,
            generated_answer=response.answer,
            relevant_documents=case.relevant_documents,
            retrieved_sources=retrieved_sources,
            recall_at_k=RetrievalMetricsCalculator.recall_at_k(
                retrieved_documents,
                case.relevant_documents,
                self._top_k,
            ),
            reciprocal_rank=RetrievalMetricsCalculator.reciprocal_rank(
                retrieved_documents,
                case.relevant_documents,
            ),
            latency_ms=response.latency_ms,
        )

    def _calculate_metrics(
        self,
        case_results: Sequence[ExperimentCaseResult],
    ) -> ExperimentMetrics:
        case_count = len(case_results)
        mean_recall = sum(result.recall_at_k for result in case_results) / case_count
        mean_reciprocal_rank = (
            sum(result.reciprocal_rank for result in case_results) / case_count
        )
        mean_latency_ms = sum(result.latency_ms for result in case_results) / case_count

        return ExperimentMetrics(
            case_count=case_count,
            top_k=self._top_k,
            mean_recall_at_k=mean_recall,
            mean_reciprocal_rank=mean_reciprocal_rank,
            mean_latency_ms=mean_latency_ms,
        )

    @staticmethod
    def _build_run_id(experiment_id: str, execution_time: datetime) -> str:
        timestamp = execution_time.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}_{experiment_id}"

    @staticmethod
    def _current_utc_time() -> datetime:
        return datetime.now(UTC)
