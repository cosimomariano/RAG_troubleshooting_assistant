import json

import httpx
import pytest

from app.generation import (
    LLMClient,
    LLMResponseError,
    LLMServiceUnavailableError,
    OllamaLLMClient,
)

OLLAMA_URL = "http://ollama-test:11434"
OLLAMA_MODEL = "modello-test"
TIMEOUT_SECONDS = 30


def build_client(transport: httpx.MockTransport) -> tuple[OllamaLLMClient, httpx.Client]:
    """Crea un client Ollama che usa un trasporto HTTP simulato."""

    http_client = httpx.Client(transport=transport)
    ollama_client = OllamaLLMClient(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        timeout_seconds=TIMEOUT_SECONDS,
        http_client=http_client,
    )
    return ollama_client, http_client


def test_generate_sends_the_prompt_and_returns_ollama_text() -> None:
    received_request: httpx.Request | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal received_request
        received_request = request
        return httpx.Response(
            200,
            json={
                "response": "  Verificare la disponibilità del servizio payment.  ",
                "done": True,
            },
        )

    ollama_client, http_client = build_client(httpx.MockTransport(handle_request))
    with http_client:
        answer = ollama_client.generate("Analizza il timeout del servizio checkout.")

    assert isinstance(ollama_client, LLMClient)
    assert answer == "Verificare la disponibilità del servizio payment."
    assert received_request is not None
    assert str(received_request.url) == f"{OLLAMA_URL}/api/generate"
    assert json.loads(received_request.content) == {
        "model": OLLAMA_MODEL,
        "prompt": "Analizza il timeout del servizio checkout.",
        "stream": False,
    }


def test_timeout_is_reported_as_unavailable_service() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout simulato", request=request)

    ollama_client, http_client = build_client(httpx.MockTransport(handle_request))
    with http_client, pytest.raises(LLMServiceUnavailableError, match="timeout"):
        ollama_client.generate("Analizza l'incidente.")


def test_connection_error_is_reported_as_unavailable_service() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connessione simulata non disponibile", request=request)

    ollama_client, http_client = build_client(httpx.MockTransport(handle_request))
    with (
        http_client,
        pytest.raises(
            LLMServiceUnavailableError,
            match="Il servizio Ollama remoto è attualmente non raggiungibile",
        ),
    ):
        ollama_client.generate("Analizza l'incidente.")


def test_http_error_preserves_the_remote_status_code() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model unavailable"})

    ollama_client, http_client = build_client(httpx.MockTransport(handle_request))
    with http_client, pytest.raises(LLMResponseError, match="503"):
        ollama_client.generate("Analizza l'incidente.")


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            httpx.Response(200, text="contenuto non JSON"),
            id="json-non-valido",
        ),
        pytest.param(
            httpx.Response(200, json={"done": True}),
            id="campo-response-assente",
        ),
        pytest.param(
            httpx.Response(200, json={"response": "   ", "done": True}),
            id="risposta-vuota",
        ),
    ],
)
def test_malformed_ollama_response_is_rejected(response: httpx.Response) -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        return response

    ollama_client, http_client = build_client(httpx.MockTransport(handle_request))
    with http_client, pytest.raises(LLMResponseError):
        ollama_client.generate("Analizza l'incidente.")


@pytest.mark.parametrize(
    ("base_url", "model", "timeout_seconds"),
    [
        pytest.param("ollama-test:11434", OLLAMA_MODEL, 30, id="url-senza-schema"),
        pytest.param(OLLAMA_URL, "   ", 30, id="modello-vuoto"),
        pytest.param(OLLAMA_URL, OLLAMA_MODEL, 0, id="timeout-zero"),
        pytest.param(OLLAMA_URL, OLLAMA_MODEL, -1, id="timeout-negativo"),
    ],
)
def test_invalid_configuration_is_rejected(
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        OllamaLLMClient(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )


def test_empty_prompt_is_rejected_before_the_http_call() -> None:
    request_sent = False

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal request_sent
        request_sent = True
        return httpx.Response(200, json={"response": "Risposta non attesa"})

    ollama_client, http_client = build_client(httpx.MockTransport(handle_request))
    with http_client, pytest.raises(ValueError, match="Prompt mancante"):
        ollama_client.generate("   ")

    assert request_sent is False
