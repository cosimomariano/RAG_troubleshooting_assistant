"""Stub del client LLM (per testare prima di sviluppare la parte di Ollama)"""

class StubLLMClient:
    def __init__(self, response: str) -> None:
        normalized_response = response.strip()
        if not normalized_response:
            raise ValueError("La risposta dello stub non può essere vuota.")

        self.response = normalized_response
        self.received_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Il prompt non può essere vuoto.")

        self.received_prompts.append(prompt)
        return self.response