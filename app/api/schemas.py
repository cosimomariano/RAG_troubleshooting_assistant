from datetime import datetime
from typing import Literal

from pydantic import Field

from app.models import OperationalMetrics, RAGResponse, SourceReference, StrictModel


class LogEvidence(StrictModel):
    timestamp: datetime | None = Field(
        default=None,
        description="Timestamp del log, se disponibile.",
    )
    service: str = Field(
        min_length=1,
        description="Microservizio che ha prodotto il log.",
    )
    severity: str | None = Field(
        default=None,
        description="Livello di severità del log.",
    )
    message: str = Field(
        min_length=1,
        description="Messaggio di log normalizzato.",
    )
    errorType: str | None = Field(
        default=None,
        description="Tipo di errore o eccezione, se disponibile.",
    )


class SpanEvidence(StrictModel):
    spanId: str | None = Field(
        default=None,
        description="Identificativo dello span, se disponibile.",
    )
    service: str = Field(
        min_length=1,
        description="Microservizio associato allo span.",
    )
    operation: str = Field(
        min_length=1,
        description="Operazione rappresentata dallo span.",
    )
    status: Literal["OK", "ERROR", "UNSET"] = Field(
        description="Stato normalizzato dello span.",
    )
    peerService: str | None = Field(
        default=None,
        description="Servizio remoto coinvolto nell'operazione, se disponibile.",
    )
    errorMessage: str | None = Field(
        default=None,
        description="Messaggio di errore associato allo span, se disponibile.",
    )


class MetricEvidence(StrictModel):
    service: str | None = Field(
        default=None,
        description="Microservizio al quale la metrica è associata.",
    )
    name: str = Field(
        min_length=1,
        description="Nome della metrica.",
    )
    value: float = Field(description="Valore osservato della metrica.")
    unit: str | None = Field(
        default=None,
        description="Unità di misura della metrica, se disponibile.",
    )


class TelemetryContext(StrictModel):
    traceId: str | None = Field(
        default=None,
        description="Identificativo della trace associata all'incidente.",
    )
    logs: list[LogEvidence] = Field(
        default_factory=list,
        description="Evidenze provenienti dai log applicativi.",
    )
    spans: list[SpanEvidence] = Field(
        default_factory=list,
        description="Span rilevanti estratti dalla trace distribuita.",
    )
    metrics: list[MetricEvidence] = Field(
        default_factory=list,
        description="Metriche considerate rilevanti per l'incidente.",
    )


class TroubleshootingRequest(StrictModel):
    question: str = Field(
        min_length=1,
        description="Domanda tecnica alla quale l'assistente deve rispondere.",
    )
    incidentContext: str | None = Field(
        default=None,
        description="Descrizione testuale opzionale del contesto dell'incidente.",
    )
    service: str | None = Field(
        default=None,
        description="Microservizio principalmente interessato, se noto.",
    )
    telemetry: TelemetryContext | None = Field(
        default=None,
        description="Evidenze di telemetria normalizzate relative all'incidente.",
    )


class TroubleshootingResponse(StrictModel):
    answer: str = Field(
        min_length=1,
        description="Risposta generata sulla base delle evidenze recuperate.",
    )
    sources: list[SourceReference] = Field(
        description="Fonti documentali utilizzate per costruire la risposta.",
    )
    latencyMs: float = Field(
        ge=0,
        description="Tempo totale di elaborazione espresso in millisecondi.",
    )
    operationalMetrics: OperationalMetrics = Field(
        description="Dettaglio delle latenze operative e dell'utilizzo dei token.",
    )

    @classmethod
    def fromRagResponse(cls, response: RAGResponse) -> "TroubleshootingResponse":
        """Converte il risultato interno del RAG nel body pubblico dell'API."""

        return cls.model_validate(response.model_dump())


class ErrorResponse(StrictModel):
    code: str = Field(
        min_length=1,
        description="Codice applicativo stabile che identifica l'errore.",
    )
    message: str = Field(
        min_length=1,
        description="Descrizione dell'errore in forma leggibile.",
    )
