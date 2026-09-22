class StubLLMClient:
    def __init__(self, response: str) -> None:
        normalized_response = response.strip()
        if not normalized_response:
            raise ValueError("La risposta dello stub non può essere vuota.")

        self._response = normalized_response
        self.received_prompts: list[str] = []

    @property
    def response(self) -> str:
        return self._response

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Il prompt non può essere vuoto.")

        self.received_prompts.append(prompt)
        return self._response
