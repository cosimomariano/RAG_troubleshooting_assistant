from app.models import LLMGenerationResult, TokenUsage


class StubLLMClient:
    def __init__(
        self,
        response: str,
        tokenUsage: TokenUsage | None = None,
    ) -> None:
        normalizedResponse = response.strip()
        if not normalizedResponse:
            raise ValueError("La risposta dello stub non può essere vuota.")

        self.response = normalizedResponse
        self.tokenUsage = tokenUsage
        self.receivedPrompts: list[str] = []

    def generate(self, prompt: str) -> str:
        return self.generateWithMetrics(prompt).text

    def generateWithMetrics(self, prompt: str) -> LLMGenerationResult:
        if not prompt.strip():
            raise ValueError("Il prompt non può essere vuoto.")

        self.receivedPrompts.append(prompt)
        return LLMGenerationResult(
            text=self.response,
            tokenUsage=self.tokenUsage,
        )
