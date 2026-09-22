import httpx
from app.generation.errors import LLMResponseError, LLMServiceUnavailableError

class OllamaLLMClient:
    # Client HTTP di ollama al quale viene inviato il prompt e ricevuta la risposta

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
        self._http_client = http_client if http_client is not None else httpx.Client()

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("Prompt mancante")

        request_body = {
            "model": self._model,
            "prompt": normalized_prompt,
            "stream": False,
        }

        response = self._send_request(request_body)
        return self._extract_generated_text(response)

    def _send_request(self, request_body: dict[str, object]) -> httpx.Response:
        endpoint = f"{self._base_url}{self.GENERATE_PATH}"

        try:
            response = self._http_client.post(
                endpoint,
                json=request_body,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMServiceUnavailableError(
                "Il servizio Ollama non ha risposto entro il timeout configurato"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMServiceUnavailableError(
                "Il servizio Ollama remoto è attualmente non raggiungibile"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMResponseError(
                f"Il servizio Ollama ha restituito il seguente stato HTTP: {exc.response.status_code}."
            ) from exc

        return response

    @staticmethod
    def _extract_generated_text(response: httpx.Response) -> str:
        try:
            response_body = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "Il servizio Ollama ha restituito un corpo che non contiene JSON valido."
            ) from exc

        if not isinstance(response_body, dict):
            raise LLMResponseError(
                "Il servizio Ollama ha restituito una struttura JSON non valida."
            )

        generated_text = response_body.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise LLMResponseError("La risposta di Ollama non contiene il testo generato.")

        return generated_text.strip()


# Metodi di validazione
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