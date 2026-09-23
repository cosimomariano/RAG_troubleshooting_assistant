import re

_TECHNICAL_TOKEN_PATTERN = re.compile(r"\w+(?:[./:-]\w+)*", re.UNICODE)

class TechnicalTextTokenizer:
    """Estrae parole e identificativi tecnici mantenendone i separatori interni."""

    def tokenize(self, text: str) -> list[str]:
        normalized_text = text.casefold()
        return _TECHNICAL_TOKEN_PATTERN.findall(normalized_text)