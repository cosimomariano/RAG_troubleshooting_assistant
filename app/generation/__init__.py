"""Componenti per la costruzione del prompt e la generazione della risposta."""

from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder
from app.generation.stub import StubLLMClient

__all__ = ["LLMClient", "PromptBuilder", "StubLLMClient"]
