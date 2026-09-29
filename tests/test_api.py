from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.api import createApp
from app.generation import LLMResponseError, LLMServiceUnavailableError
from app.models import OperationalMetrics, RAGResponse, SourceReference, TokenUsage
from app.services import TroubleshootingSystem

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingTroubleshootingService:
    """Servizio controllato che registra i dati ricevuti dal controller."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def troubleshoot(
        self,
        question: str,
        incidentContext: str | None = None,
    ) -> RAGResponse:
        self.calls.append((question, incidentContext))
        return RAGResponse(
            answer="Il servizio payment non è raggiungibile dal checkout.",
            sources=[
                SourceReference(
                    citationId="FONTE_1",
                    documentId="document-payment-unreachable",
                    source="runbooks/payment-unreachable.md",
                    chunkId="payment-unreachable-001",
                    documentType="runbook",
                    section="Diagnosi",
                    service="payment",
                    category="service-unavailable",
                    rank=1,
                    retriever="hybrid",
                    fusedScore=0.0328,
                    rerankerScore=4.82,
                )
            ],
            latencyMs=18.5,
            operationalMetrics=OperationalMetrics(
                retrievalLatencyMs=3.0,
                rerankingLatencyMs=1.5,
                promptBuildLatencyMs=0.5,
                generationLatencyMs=12.0,
                totalLatencyMs=18.5,
                tokenUsage=TokenUsage(
                    inputTokens=120,
                    outputTokens=30,
                    totalTokens=150,
                ),
            ),
        )


class FailingTroubleshootingService:
    def __init__(self, exception: Exception) -> None:
        self.exception = exception

    def troubleshoot(
        self,
        question: str,
        incidentContext: str | None = None,
    ) -> RAGResponse:
        raise self.exception


def testApiAcceptsNormalizedTelemetryAndReturnsSources() -> None:
    service = RecordingTroubleshootingService()
    application = createApp(service)

    with TestClient(application) as client:
        response = client.post(
            "/troubleshoot",
            json={
                "question": "Perché il checkout non completa il pagamento?",
                "incident_context": "La fase di charge non viene completata.",
                "service": "checkout",
                "telemetry": {
                    "trace_id": "trace-example-001",
                    "logs": [
                        {
                            "service": "checkout",
                            "severity": "ERROR",
                            "message": "connection refused while contacting payment",
                            "error_type": "ConnectionRefusedError",
                        }
                    ],
                    "spans": [
                        {
                            "span_id": "span-001",
                            "service": "checkout",
                            "operation": "payment/charge",
                            "status": "ERROR",
                            "peer_service": "payment",
                        }
                    ],
                    "metrics": [
                        {
                            "service": "checkout",
                            "name": "request.duration",
                            "value": 2500,
                            "unit": "ms",
                        }
                    ],
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Il servizio payment non è raggiungibile dal checkout.",
        "sources": [
            {
                "citation_id": "FONTE_1",
                "document_id": "document-payment-unreachable",
                "source": "runbooks/payment-unreachable.md",
                "chunk_id": "payment-unreachable-001",
                "document_type": "runbook",
                "section": "Diagnosi",
                "service": "payment",
                "category": "service-unavailable",
                "rank": 1,
                "retriever": "hybrid",
                "score": None,
                "fused_score": 0.0328,
                "reranker_score": 4.82,
            }
        ],
        "latency_ms": 18.5,
        "operational_metrics": {
            "retrieval_latency_ms": 3.0,
            "reranking_latency_ms": 1.5,
            "prompt_build_latency_ms": 0.5,
            "generation_latency_ms": 12.0,
            "total_latency_ms": 18.5,
            "token_usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            },
        },
    }

    question, incidentContext = service.calls[0]
    assert question == "Perché il checkout non completa il pagamento?"
    assert incidentContext is not None
    assert "Contesto dichiarato: La fase di charge non viene completata." in incidentContext
    assert "Servizio principale: checkout" in incidentContext
    assert "Trace ID: trace-example-001" in incidentContext
    assert "Log: servizio=checkout" in incidentContext
    assert "Span: servizio=checkout" in incidentContext
    assert "Metrica: nome=request.duration" in incidentContext


def testOptionalIncidentDataCanBeOmitted() -> None:
    service = RecordingTroubleshootingService()

    with TestClient(createApp(service)) as client:
        response = client.post(
            "/troubleshoot",
            json={"question": "Qual è la causa dell'errore?"},
        )

    assert response.status_code == 200
    assert service.calls == [("Qual è la causa dell'errore?", None)]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="domanda-assente"),
        pytest.param(
            {"question": "Domanda valida", "campo_sconosciuto": True},
            id="campo-non-previsto",
        ),
        pytest.param(
            {
                "question": "Domanda valida",
                "telemetry": {
                    "spans": [
                        {
                            "service": "checkout",
                            "operation": "payment/charge",
                            "status": "INVALID",
                        }
                    ]
                },
            },
            id="stato-span-non-valido",
        ),
    ],
)
def testInvalidBodyReturnsTheContractError(body: dict[str, object]) -> None:
    with TestClient(createApp(RecordingTroubleshootingService())) as client:
        response = client.post("/troubleshoot", json=body)

    assert response.status_code == 422
    assert response.json() == {
        "code": "VALIDATION_ERROR",
        "message": "Il corpo della richiesta non rispetta il contratto API previsto.",
    }


@pytest.mark.parametrize(
    ("exception", "expectedStatus", "expectedBody"),
    [
        pytest.param(
            ValueError("Domanda vuota"),
            400,
            {"code": "INVALID_REQUEST", "message": "La richiesta non può essere elaborata."},
            id="richiesta-non-valida",
        ),
        pytest.param(
            LLMResponseError("Risposta remota non valida"),
            502,
            {
                "code": "LLM_INVALID_RESPONSE",
                "message": "Il servizio LLM remoto ha restituito una risposta non valida.",
            },
            id="risposta-llm-non-valida",
        ),
        pytest.param(
            LLMServiceUnavailableError("Timeout simulato"),
            503,
            {
                "code": "LLM_SERVICE_UNAVAILABLE",
                "message": "Il servizio LLM remoto non è disponibile.",
            },
            id="llm-non-disponibile",
        ),
    ],
)
def testKnownApplicationErrorsAreMappedToContractResponses(
    exception: Exception,
    expectedStatus: int,
    expectedBody: dict[str, str],
) -> None:
    application = createApp(FailingTroubleshootingService(exception))

    with TestClient(application) as client:
        response = client.post("/troubleshoot", json={"question": "Analizza l'incidente"})

    assert response.status_code == expectedStatus
    assert response.json() == expectedBody


def testUnexpectedErrorReturnsAGenericInternalError() -> None:
    application = createApp(FailingTroubleshootingService(RuntimeError("Errore simulato")))

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/troubleshoot", json={"question": "Analizza l'incidente"})

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "Errore interno durante l'elaborazione della richiesta.",
    }


def testRagServiceShapeSatisfiesTheControllerContract() -> None:
    service = RecordingTroubleshootingService()

    assert isinstance(service, TroubleshootingSystem)


def testGeneratedOpenapiMatchesThePublicContractFields() -> None:
    contractPath = PROJECT_ROOT / "contracts/openapi/troubleshooting-api.yaml"
    publicContract = yaml.safe_load(contractPath.read_text(encoding="utf-8"))
    generatedContract = createApp(RecordingTroubleshootingService()).openapi()

    expectedSchemas = publicContract["components"]["schemas"]
    generatedSchemas = generatedContract["components"]["schemas"]

    for schemaName in [
        "TroubleshootingRequest",
        "TelemetryContext",
        "LogEvidence",
        "SpanEvidence",
        "MetricEvidence",
        "TroubleshootingResponse",
        "OperationalMetrics",
        "TokenUsage",
        "SourceReference",
        "ErrorResponse",
    ]:
        assert set(generatedSchemas[schemaName]["properties"]) == set(
            expectedSchemas[schemaName]["properties"]
        )
        assert set(generatedSchemas[schemaName].get("required", [])) == set(
            expectedSchemas[schemaName].get("required", [])
        )
        assert generatedSchemas[schemaName]["additionalProperties"] is False

    generatedOperation = generatedContract["paths"]["/troubleshoot"]["post"]
    expectedOperation = publicContract["paths"]["/troubleshoot"]["post"]

    assert generatedOperation["operationId"] == expectedOperation["operationId"]
    assert set(generatedOperation["responses"]) == set(expectedOperation["responses"])
