class CitationFormatter:
    """Costruisce identificativi coerenti per prompt e risposta API."""

    PREFIX = "FONTE"

    @staticmethod
    def build_identifier(position: int) -> str:
        if position < 1:
            raise ValueError("La posizione della citazione deve essere maggiore di zero.")
        return f"{CitationFormatter.PREFIX}_{position}"

    @staticmethod
    def format_reference(position: int) -> str:
        identifier = CitationFormatter.build_identifier(position)
        return f"[{identifier}]"
