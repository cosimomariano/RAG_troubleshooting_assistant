from abc import ABC, abstractmethod

class SensitiveDataMasker(ABC):
    """Contratto per il mascheramento dei dati sensibili presenti nel testo."""

    @abstractmethod
    def mask(self, text: str) -> str:
        """Restituisce il testo dopo aver sostituito i valori sensibili rilevati."""