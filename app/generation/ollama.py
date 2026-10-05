import httpx

from app.generation.errors import LLMResponseError, LLMServiceUnavailableError
from app.models import LLMGenerationResult, TokenUsage

OllamaRequestBody = dict[str, object]


class OllamaLLMClient:
    GENERATE_PATH = "/api/generate"

    def __init__(
        self,
        baseUrl: str,
        model: str,
        timeoutSeconds: float,
        *,
        httpClient: httpx.Client | None = None,
    ) -> None:
        self.baseUrl = self.validateBaseUrl(baseUrl)
        self.model = self.validateModel(model)
        self.timeoutSeconds = self.validateTimeout(timeoutSeconds)
        self.httpClient = self.resolveHttpClient(httpClient)

    def generate(self, prompt: str) -> str:
        return self.generateWithMetrics(prompt).text

    def generateWithMetrics(self, prompt: str) -> LLMGenerationResult:
        # Normalizzazione del prompt ed incapsulamento in oggetto di request
        normalizedPrompt = self.normalizePrompt(prompt)
        requestBody = self.buildRequestBody(normalizedPrompt)

        # Sottomissione della richiesta e recupero della risposta
        response = self.sendRequest(requestBody)
        return self.extractGenerationResult(response)

    def buildRequestBody(self, prompt: str) -> OllamaRequestBody:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

    def sendRequest(self, requestBody: OllamaRequestBody) -> httpx.Response:
        endpoint = f"{self.baseUrl}{self.GENERATE_PATH}"

        try:
            response = self.httpClient.post(
                endpoint,
                json=requestBody,
                timeout=self.timeoutSeconds,
            )
            response.raise_for_status()
            return response
        except httpx.TimeoutException as error:
            raise LLMServiceUnavailableError(
                "Il servizio Ollama non ha risposto entro il timeout configurato"
            ) from error
        except httpx.RequestError as error:
            raise LLMServiceUnavailableError(
                "Il servizio Ollama remoto è attualmente non raggiungibile"
            ) from error
        except httpx.HTTPStatusError as error:
            statusCode = error.response.status_code
            if statusCode == 429 or statusCode >= 500:
                raise LLMServiceUnavailableError(
                    "Il servizio Ollama remoto non è temporaneamente disponibile."
                ) from error
            raise LLMResponseError(
                f"Il servizio Ollama ha restituito il seguente stato HTTP: {statusCode}."
            ) from error

    @staticmethod
    def extractGenerationResult(response: httpx.Response) -> LLMGenerationResult:
        responseBody = OllamaLLMClient.readResponseBody(response)
        generatedText = responseBody.get("response")

        if not isinstance(generatedText, str) or not generatedText.strip():
            raise LLMResponseError("La risposta di Ollama non contiene il testo generato.")
        return LLMGenerationResult(
            text=generatedText,
            tokenUsage=OllamaLLMClient.extractTokenUsage(responseBody),
        )

    @staticmethod
    def extractTokenUsage(responseBody: dict[str, object]) -> TokenUsage | None:
        inputTokens = responseBody.get("prompt_eval_count")
        outputTokens = responseBody.get("eval_count")

        if inputTokens is None and outputTokens is None:
            return None
        if inputTokens is None or outputTokens is None:
            raise LLMResponseError("La risposta di Ollama contiene un conteggio token incompleto.")

        validatedInputTokens = OllamaLLMClient.validateTokenCount(
            inputTokens,
            "prompt_eval_count",
        )
        validatedOutputTokens = OllamaLLMClient.validateTokenCount(
            outputTokens,
            "eval_count",
        )
        return TokenUsage(
            inputTokens=validatedInputTokens,
            outputTokens=validatedOutputTokens,
            totalTokens=validatedInputTokens + validatedOutputTokens,
        )

    @staticmethod
    def validateTokenCount(value: object, fieldName: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LLMResponseError(f"Il campo {fieldName} restituito da Ollama non è valido.")
        return value

    @staticmethod
    def readResponseBody(response: httpx.Response) -> dict[str, object]:
        try:
            responseBody = response.json()
        except ValueError as error:
            raise LLMResponseError(
                "Il servizio Ollama ha restituito un corpo che non contiene JSON valido."
            ) from error

        if not isinstance(responseBody, dict):
            raise LLMResponseError(
                "Il servizio Ollama ha restituito una struttura JSON non valida."
            )
        return responseBody

    @staticmethod
    def resolveHttpClient(httpClient: httpx.Client | None) -> httpx.Client:
        if httpClient is not None:
            return httpClient
        return httpx.Client()

    @staticmethod
    def normalizePrompt(prompt: str) -> str:
        normalizedPrompt = prompt.strip()
        if not normalizedPrompt:
            raise ValueError("Prompt mancante")
        return normalizedPrompt

    @staticmethod
    def validateBaseUrl(baseUrl: str) -> str:
        normalizedUrl = baseUrl.strip().rstrip("/")
        parsedUrl = httpx.URL(normalizedUrl)

        if parsedUrl.scheme not in {"http", "https"} or not parsedUrl.host:
            raise ValueError("L'URL di Ollama deve essere un indirizzo HTTP o HTTPS valido.")
        return normalizedUrl

    @staticmethod
    def validateModel(model: str) -> str:
        normalizedModel = model.strip()
        if not normalizedModel:
            raise ValueError("Il nome del modello Ollama non può essere vuoto.")
        return normalizedModel

    @staticmethod
    def validateTimeout(timeoutSeconds: float) -> float:
        if timeoutSeconds <= 0:
            raise ValueError("Il timeout di Ollama deve essere maggiore di zero.")
        return float(timeoutSeconds)
