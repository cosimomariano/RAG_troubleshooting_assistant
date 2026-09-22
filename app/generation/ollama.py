import httpx

from app.generation.errors import LLMResponseError, LLMServiceUnavailableError

OllamaRequestBody = dict[str, object]


class OllamaLLMClient:
    GENERATE_PATH = "/api/generate"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = self._validate_base_url(base_url)
        self._model = self._validate_model(model)
        self._timeout_seconds = self._validate_timeout(timeout_seconds)
        self._http_client = self._resolve_http_client(http_client)

    def generate(self, prompt: str) -> str:
        normalized_prompt = self._normalize_prompt(prompt)
        request_body = self._build_request_body(normalized_prompt)
        response = self._send_request(request_body)
        return self._extract_generated_text(response)

    def _build_request_body(self, prompt: str) -> OllamaRequestBody:
        return {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

    def _send_request(self, request_body: OllamaRequestBody) -> httpx.Response:
        endpoint = f"{self._base_url}{self.GENERATE_PATH}"

        try:
            response = self._http_client.post(
                endpoint,
                json=request_body,
                timeout=self._timeout_seconds,
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
            status_code = error.response.status_code
            raise LLMResponseError(
                f"Il servizio Ollama ha restituito il seguente stato HTTP: {status_code}."
            ) from error

    @staticmethod
    def _extract_generated_text(response: httpx.Response) -> str:
        response_body = OllamaLLMClient._read_response_body(response)
        generated_text = response_body.get("response")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise LLMResponseError("La risposta di Ollama non contiene il testo generato.")
        return generated_text.strip()

    @staticmethod
    def _read_response_body(response: httpx.Response) -> dict[str, object]:
        try:
            response_body = response.json()
        except ValueError as error:
            raise LLMResponseError(
                "Il servizio Ollama ha restituito un corpo che non contiene JSON valido."
            ) from error

        if not isinstance(response_body, dict):
            raise LLMResponseError(
                "Il servizio Ollama ha restituito una struttura JSON non valida."
            )
        return response_body

    @staticmethod
    def _resolve_http_client(http_client: httpx.Client | None) -> httpx.Client:
        if http_client is not None:
            return http_client
        return httpx.Client()

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("Prompt mancante")
        return normalized_prompt

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        normalized_url = base_url.strip().rstrip("/")
        parsed_url = httpx.URL(normalized_url)

        if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
            raise ValueError("L'URL di Ollama deve essere un indirizzo HTTP o HTTPS valido.")
        return normalized_url

    @staticmethod
    def _validate_model(model: str) -> str:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("Il nome del modello Ollama non può essere vuoto.")
        return normalized_model

    @staticmethod
    def _validate_timeout(timeout_seconds: float) -> float:
        if timeout_seconds <= 0:
            raise ValueError("Il timeout di Ollama deve essere maggiore di zero.")
        return float(timeout_seconds)
