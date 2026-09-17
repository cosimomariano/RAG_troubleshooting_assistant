import re
import pytest
from app.ingestion import MaskingRule, RegexSensitiveDataMasker, SensitiveDataMasker

@pytest.fixture
def masker() -> RegexSensitiveDataMasker:
    return RegexSensitiveDataMasker()

def test_default_masker_implements_masking_contract(
    masker: RegexSensitiveDataMasker,
) -> None:
    assert isinstance(masker, SensitiveDataMasker)

def test_email_is_masked(masker: RegexSensitiveDataMasker) -> None:
    text = "Contattare mario.rossi@example.test per ulteriori dettagli."

    masked_text = masker.mask(text)

    assert masked_text == "Contattare [MASCHERATO:EMAIL] per ulteriori dettagli."

@pytest.mark.parametrize(
    ("text", "expected_text"),
    [
        (
            "api_key=test-token-123456",
            "api_key=[MASCHERATO:CREDENZIALE]",
        ),
        (
            "Authorization: Bearer synthetic.token.123456",
            "Authorization: Bearer [MASCHERATO:CREDENZIALE]",
        ),
    ],
)
def test_token_like_credentials_are_masked(
    masker: RegexSensitiveDataMasker,
    text: str,
    expected_text: str,
) -> None:
    assert masker.mask(text) == expected_text

def test_iban_like_value_is_masked(masker: RegexSensitiveDataMasker) -> None:
    text = "IBAN di test: IT60 X054 2811 1010 0000 0123 456."

    masked_text = masker.mask(text)

    assert masked_text == "IBAN di test: [MASCHERATO:IBAN]."

def test_private_ipv4_is_masked_and_public_ipv4_is_preserved(
    masker: RegexSensitiveDataMasker,
) -> None:
    text = "Servizio interno 10.23.4.5, resolver pubblico 8.8.8.8."

    masked_text = masker.mask(text)

    assert masked_text == (
        "Servizio interno [MASCHERATO:IP_PRIVATO], resolver pubblico 8.8.8.8."
    )

def test_masking_is_deterministic_and_idempotent(masker: RegexSensitiveDataMasker) -> None:
    text = "Email admin@example.test e password=synthetic-secret-123."

    first_result = masker.mask(text)

    assert masker.mask(text) == first_result
    assert masker.mask(first_result) == first_result

def test_custom_rule_can_extend_the_masker() -> None:
    custom_rule = MaskingRule(
        name="synthetic_ticket",
        pattern=re.compile(r"TICKET-\d{4}"),
        replacement="[MASCHERATO:TICKET]",
    )
    masker = RegexSensitiveDataMasker(rules=[custom_rule])

    assert masker.mask("Riferimento TICKET-1234") == "Riferimento [MASCHERATO:TICKET]"

@pytest.mark.parametrize("text", ["", "Testo senza dati sensibili."])
def test_text_without_sensitive_data_is_unchanged(
    masker: RegexSensitiveDataMasker,
    text: str,
) -> None:
    assert masker.mask(text) == text