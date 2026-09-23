import re

_TECHNICAL_TOKEN_PATTERN = re.compile(r"\w+(?:[./:-]\w+)*", re.UNICODE)

class TechnicalTextTokenizer:
    """Estrae parole e identificativi tecnici mantenendone i separatori interni."""

    def tokenize(self, text: str) -> list[str]:
        # Rende tutto il testo lower case
        normalized_text = text.casefold()
        # Estrae i token (es. una frase di 5 parole torna una lista di 5 elementi di tipo stringa)
        return _TECHNICAL_TOKEN_PATTERN.findall(normalized_text)