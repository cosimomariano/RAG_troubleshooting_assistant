import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
from app.ingestion.masking.base import SensitiveDataMasker

 # Questa classe estende la classe base e viene richiamata da quest'ultima per le operazioni di mascheramento

MaskingReplacement = str | Callable[[re.Match[str]], str]

_CREDENTIAL_PLACEHOLDER = "[MASCHERATO:CREDENZIALE]"
_EMAIL_PLACEHOLDER = "[MASCHERATO:EMAIL]"
_IBAN_PLACEHOLDER = "[MASCHERATO:IBAN]"
_PRIVATE_IPV4_PLACEHOLDER = "[MASCHERATO:IP_PRIVATO]"

_PRIVATE_IPV4_NETWORKS = (
    IPv4Network("10.0.0.0/8"),
    IPv4Network("172.16.0.0/12"),
    IPv4Network("192.168.0.0/16"),
)

@dataclass(frozen=True, slots=True)
class MaskingRule:
    """Regex per il replace dei dati sensibili"""

    name: str
    pattern: re.Pattern[str]
    replacement: MaskingReplacement

def _mask_labeled_credential(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}{_CREDENTIAL_PLACEHOLDER}"

def _mask_bearer_token(match: re.Match[str]) -> str:
    return f"{match.group(1)}{_CREDENTIAL_PLACEHOLDER}"

def _mask_private_ipv4(match: re.Match[str]) -> str:
    address = IPv4Address(match.group(0))
    if any(address in network for network in _PRIVATE_IPV4_NETWORKS):
        return _PRIVATE_IPV4_PLACEHOLDER
    return match.group(0)

DEFAULT_MASKING_RULES = (
    MaskingRule(
        name="labeled_credential",
        pattern=re.compile(
            r"\b(api[_-]?key|access[_-]?token|token|secret|password)\b(\s*[:=]\s*)"
            r"[^\s,;]+",
            re.IGNORECASE,
        ),
        replacement=_mask_labeled_credential,
    ),
    MaskingRule(
        name="bearer_token",
        pattern=re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
        replacement=_mask_bearer_token,
    ),
    MaskingRule(
        name="email",
        pattern=re.compile(
            r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])"
        ),
        replacement=_EMAIL_PLACEHOLDER,
    ),
    MaskingRule(
        name="iban",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.IGNORECASE),
        replacement=_IBAN_PLACEHOLDER,
    ),
    MaskingRule(
        name="private_ipv4",
        pattern=re.compile(
            r"(?<![\d.])"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}"
            r"(?!\d)(?!\.\d)"
        ),
        replacement=_mask_private_ipv4,
    ),
)

class RegexSensitiveDataMasker(SensitiveDataMasker):
    def __init__(self, rules: Iterable[MaskingRule] | None = None) -> None:
        self._rules = tuple(rules) if rules is not None else DEFAULT_MASKING_RULES

    def mask(self, text: str) -> str:
        masked_text = text
        for rule in self._rules:
            masked_text = rule.pattern.sub(rule.replacement, masked_text)
        return masked_text