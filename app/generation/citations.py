class CitationFormatter:
    """Costruisce identificativi coerenti per prompt e risposta API."""

    PREFIX = "FONTE"

    @staticmethod
    def buildIdentifier(position: int) -> str:
        if position < 1:
            raise ValueError("La posizione della citazione deve essere maggiore di zero.")
        return f"{CitationFormatter.PREFIX}_{position}"

    @staticmethod
    def formatReference(position: int) -> str:
        identifier = CitationFormatter.buildIdentifier(position)
        return f"[{identifier}]"
