from app.models import LLMGenerationResult, TokenUsage


class StubLLMClient:
    def __init__(
        self,
        response: str,
        token_usage: TokenUsage | None = None,
    ) -> None:
        normalized_response = response.strip()
        if not normalized_response:
            raise ValueError("La risposta dello stub non può essere vuota.")

        self._response = normalized_response
        self._token_usage = token_usage
        self.received_prompts: list[str] = []

    @property
    def response(self) -> str:
        return self._response

    def generate(self, prompt: str) -> str:
        return self.generate_with_metrics(prompt).text

    def generate_with_metrics(self, prompt: str) -> LLMGenerationResult:
        if not prompt.strip():
            raise ValueError("Il prompt non può essere vuoto.")

        self.received_prompts.append(prompt)
        return LLMGenerationResult(
            text=self._response,
            token_usage=self._token_usage,
        )
