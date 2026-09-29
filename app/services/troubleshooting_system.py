from typing import Protocol, runtime_checkable

from app.models import RAGResponse


@runtime_checkable
class TroubleshootingSystem(Protocol):
    def troubleshoot(
        self,
        question: str,
        incidentContext: str | None = None,
    ) -> RAGResponse: ...
