from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.api import TroubleshootingService, create_app
from app.generation import LLMResponseError, LLMServiceUnavailableError
from app.models import OperationalMetrics, RAGResponse, SourceReference, TokenUsage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingTroubleshootingService:
    """Servizio controllato che registra i dati ricevuti dal controller."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse:
        self.calls.append((question, incident_context))
        return RAGResponse(
            answer="Il servizio payment non è raggiungibile dal checkout.",
            sources=[
                SourceReference(
                    citation_id="FONTE_1",
                    document_id="document-payment-unreachable",
                    source="runbooks/payment-unreachable.md",
                    chunk_id="payment-unreachable-001",
                    document_type="runbook",
                    section="Diagnosi",
                    service="payment",
                    category="service-unavailable",
                    rank=1,
                    retriever="hybrid",
                    fused_score=0.0328,
                    reranker_score=4.82,
                )
            ],
            latency_ms=18.5,
            operational_metrics=OperationalMetrics(
                retrieval_latency_ms=3.0,
                reranking_latency_ms=1.5,
                prompt_build_latency_ms=0.5,
                generation_latency_ms=12.0,
                total_latency_ms=18.5,
                token_usage=TokenUsage(
                    input_tokens=120,
                    output_tokens=30,
                    total_tokens=150,
                ),
            ),
        )


class FailingTroubleshootingService:
    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse:
        raise self._exception


def test_api_accepts_normalized_telemetry_and_returns_sources() -> None:
    service = RecordingTroubleshootingService()
    application = create_app(service)

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

    question, incident_context = service.calls[0]
    assert question == "Perché il checkout non completa il pagamento?"
    assert incident_context is not None
    assert "Contesto dichiarato: La fase di charge non viene completata." in incident_context
    assert "Servizio principale: checkout" in incident_context
    assert "Trace ID: trace-example-001" in incident_context
    assert "Log: servizio=checkout" in incident_context
    assert "Span: servizio=checkout" in incident_context
    assert "Metrica: nome=request.duration" in incident_context


def test_optional_incident_data_can_be_omitted() -> None:
    service = RecordingTroubleshootingService()

    with TestClient(create_app(service)) as client:
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
def test_invalid_body_returns_the_contract_error(body: dict[str, object]) -> None:
    with TestClient(create_app(RecordingTroubleshootingService())) as client:
        response = client.post("/troubleshoot", json=body)

    assert response.status_code == 422
    assert response.json() == {
        "code": "VALIDATION_ERROR",
        "message": "Il corpo della richiesta non rispetta il contratto API previsto.",
    }


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_body"),
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
def test_known_application_errors_are_mapped_to_contract_responses(
    exception: Exception,
    expected_status: int,
    expected_body: dict[str, str],
) -> None:
    application = create_app(FailingTroubleshootingService(exception))

    with TestClient(application) as client:
        response = client.post("/troubleshoot", json={"question": "Analizza l'incidente"})

    assert response.status_code == expected_status
    assert response.json() == expected_body


def test_unexpected_error_returns_a_generic_internal_error() -> None:
    application = create_app(FailingTroubleshootingService(RuntimeError("Errore simulato")))

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/troubleshoot", json={"question": "Analizza l'incidente"})

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "Errore interno durante l'elaborazione della richiesta.",
    }


def test_rag_service_shape_satisfies_the_controller_contract() -> None:
    service = RecordingTroubleshootingService()

    assert isinstance(service, TroubleshootingService)


def test_generated_openapi_matches_the_public_contract_fields() -> None:
    contract_path = PROJECT_ROOT / "contracts/openapi/troubleshooting-api.yaml"
    public_contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    generated_contract = create_app(RecordingTroubleshootingService()).openapi()

    expected_schemas = public_contract["components"]["schemas"]
    generated_schemas = generated_contract["components"]["schemas"]

    for schema_name in [
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
        assert set(generated_schemas[schema_name]["properties"]) == set(
            expected_schemas[schema_name]["properties"]
        )
        assert set(generated_schemas[schema_name].get("required", [])) == set(
            expected_schemas[schema_name].get("required", [])
        )
        assert generated_schemas[schema_name]["additionalProperties"] is False

    generated_operation = generated_contract["paths"]["/troubleshoot"]["post"]
    expected_operation = public_contract["paths"]["/troubleshoot"]["post"]

    assert generated_operation["operationId"] == expected_operation["operationId"]
    assert set(generated_operation["responses"]) == set(expected_operation["responses"])
