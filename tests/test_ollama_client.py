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


def buildClient(transport: httpx.MockTransport) -> tuple[OllamaLLMClient, httpx.Client]:
    """Crea un client Ollama che usa un trasporto HTTP simulato."""

    httpClient = httpx.Client(transport=transport)
    ollamaClient = OllamaLLMClient(
        baseUrl=OLLAMA_URL,
        model=OLLAMA_MODEL,
        timeoutSeconds=TIMEOUT_SECONDS,
        httpClient=httpClient,
    )
    return ollamaClient, httpClient


def testGenerateSendsThePromptAndReturnsOllamaText() -> None:
    receivedRequest: httpx.Request | None = None

    def handleRequest(request: httpx.Request) -> httpx.Response:
        nonlocal receivedRequest
        receivedRequest = request
        return httpx.Response(
            200,
            json={
                "response": "  Verificare la disponibilità del servizio payment.  ",
                "done": True,
            },
        )

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient:
        answer = ollamaClient.generate("Analizza il timeout del servizio checkout.")

    assert isinstance(ollamaClient, LLMClient)
    assert answer == "Verificare la disponibilità del servizio payment."
    assert receivedRequest is not None
    assert str(receivedRequest.url) == f"{OLLAMA_URL}/api/generate"
    assert json.loads(receivedRequest.content) == {
        "model": OLLAMA_MODEL,
        "prompt": "Analizza il timeout del servizio checkout.",
        "stream": False,
    }


def testGenerateWithMetricsMapsOllamaTokenCounts() -> None:
    def handleRequest(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": "Verificare il servizio payment.",
                "done": True,
                "prompt_eval_count": 42,
                "eval_count": 11,
            },
        )

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient:
        result = ollamaClient.generateWithMetrics("Analizza l'incidente.")

    assert isinstance(ollamaClient, LLMClient)
    assert result.text == "Verificare il servizio payment."
    assert result.tokenUsage is not None
    assert result.tokenUsage.model_dump() == {
        "input_tokens": 42,
        "output_tokens": 11,
        "total_tokens": 53,
    }


def testTimeoutIsReportedAsUnavailableService() -> None:
    def handleRequest(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout simulato", request=request)

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient, pytest.raises(LLMServiceUnavailableError, match="timeout"):
        ollamaClient.generate("Analizza l'incidente.")


def testConnectionErrorIsReportedAsUnavailableService() -> None:
    def handleRequest(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connessione simulata non disponibile", request=request)

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with (
        httpClient,
        pytest.raises(
            LLMServiceUnavailableError,
            match="Il servizio Ollama remoto è attualmente non raggiungibile",
        ),
    ):
        ollamaClient.generate("Analizza l'incidente.")


def testServerHttpErrorIsReportedAsUnavailableService() -> None:
    def handleRequest(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model unavailable"})

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient, pytest.raises(LLMServiceUnavailableError, match="temporaneamente"):
        ollamaClient.generate("Analizza l'incidente.")


def testClientHttpErrorPreservesTheRemoteStatusCode() -> None:
    def handleRequest(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid request"})

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient, pytest.raises(LLMResponseError, match="400"):
        ollamaClient.generate("Analizza l'incidente.")


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
        pytest.param(
            httpx.Response(
                200,
                json={
                    "response": "Risposta valida",
                    "prompt_eval_count": 20,
                },
            ),
            id="conteggio-token-incompleto",
        ),
        pytest.param(
            httpx.Response(
                200,
                json={
                    "response": "Risposta valida",
                    "prompt_eval_count": "venti",
                    "eval_count": 5,
                },
            ),
            id="conteggio-token-non-numerico",
        ),
    ],
)
def testMalformedOllamaResponseIsRejected(response: httpx.Response) -> None:
    def handleRequest(request: httpx.Request) -> httpx.Response:
        return response

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient, pytest.raises(LLMResponseError):
        ollamaClient.generate("Analizza l'incidente.")


@pytest.mark.parametrize(
    ("baseUrl", "model", "timeoutSeconds"),
    [
        pytest.param("ollama-test:11434", OLLAMA_MODEL, 30, id="url-senza-schema"),
        pytest.param(OLLAMA_URL, "   ", 30, id="modello-vuoto"),
        pytest.param(OLLAMA_URL, OLLAMA_MODEL, 0, id="timeout-zero"),
        pytest.param(OLLAMA_URL, OLLAMA_MODEL, -1, id="timeout-negativo"),
    ],
)
def testInvalidConfigurationIsRejected(
    baseUrl: str,
    model: str,
    timeoutSeconds: float,
) -> None:
    with pytest.raises(ValueError):
        OllamaLLMClient(
            baseUrl=baseUrl,
            model=model,
            timeoutSeconds=timeoutSeconds,
        )


def testEmptyPromptIsRejectedBeforeTheHttpCall() -> None:
    requestSent = False

    def handleRequest(request: httpx.Request) -> httpx.Response:
        nonlocal requestSent
        requestSent = True
        return httpx.Response(200, json={"response": "Risposta non attesa"})

    ollamaClient, httpClient = buildClient(httpx.MockTransport(handleRequest))
    with httpClient, pytest.raises(ValueError, match="Prompt mancante"):
        ollamaClient.generate("   ")

    assert requestSent is False
