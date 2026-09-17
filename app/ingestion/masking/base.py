from abc import ABC, abstractmethod

class SensitiveDataMasker(ABC):
    """Contratto per il mascheramento dei dati sensibili presenti nel testo."""

    @abstractmethod
    def mask(self, text: str) -> str:
        """Questo metodo astratto esegue le operazioni di mascheramento degli implementatori e torna la string mascherata"""