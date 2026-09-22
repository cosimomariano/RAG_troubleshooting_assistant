import re

import pytest

from app.ingestion import MaskingRule, RegexSensitiveDataMasker, SensitiveDataMasker


@pytest.fixture
def default_masker() -> RegexSensitiveDataMasker:
    return RegexSensitiveDataMasker()


def test_regex_masker_can_be_used_through_common_contract(
    default_masker: RegexSensitiveDataMasker,
) -> None:
    assert isinstance(default_masker, SensitiveDataMasker)


def test_on_call_email_is_removed_from_runbook_text(
    default_masker: RegexSensitiveDataMasker,
) -> None:
    runbook_line = "Contattare oncall.payment@example.test in caso di errore."
    masked_line = default_masker.mask(runbook_line)

    assert masked_line == "Contattare [MASCHERATO:EMAIL] in caso di errore."


@pytest.mark.parametrize(
    ("log_line", "expected_line"),
    [
        pytest.param(
            "api_key=demo-payment-key-123456",
            "api_key=[MASCHERATO:CREDENZIALE]",
            id="api-key-etichettata",
        ),
        pytest.param(
            "Authorization: Bearer demo.checkout.token.123456",
            "Authorization: Bearer [MASCHERATO:CREDENZIALE]",
            id="bearer-token",
        ),
    ],
)
def test_credentials_are_removed_from_telemetry_lines(
    default_masker: RegexSensitiveDataMasker,
    log_line: str,
    expected_line: str,
) -> None:
    assert default_masker.mask(log_line) == expected_line


def test_iban_like_value_is_not_sent_to_indexing(
    default_masker: RegexSensitiveDataMasker,
) -> None:
    incident_note = "IBAN usato nel test: IT60 X054 2811 1010 0000 0123 456."

    masked_note = default_masker.mask(incident_note)

    assert masked_note == "IBAN usato nel test: [MASCHERATO:IBAN]."


def test_internal_service_address_is_hidden_but_public_resolver_is_kept(
    default_masker: RegexSensitiveDataMasker,
) -> None:
    log_line = "payment risponde da 10.23.4.5; il resolver configurato è 8.8.8.8."

    masked_line = default_masker.mask(log_line)

    assert masked_line == (
        "payment risponde da [MASCHERATO:IP_PRIVATO]; il resolver configurato è 8.8.8.8."
    )


def test_private_ipv4_is_hidden_when_followed_by_sentence_period(
    default_masker: RegexSensitiveDataMasker,
) -> None:
    text = "Il servizio payment risponde da 10.23.4.5."

    assert default_masker.mask(text) == ("Il servizio payment risponde da [MASCHERATO:IP_PRIVATO].")


def test_reprocessing_masked_text_does_not_change_it_again(
    default_masker: RegexSensitiveDataMasker,
) -> None:
    log_line = "Email admin@example.test e password=demo-password-123."

    first_pass = default_masker.mask(log_line)

    assert default_masker.mask(log_line) == first_pass
    assert default_masker.mask(first_pass) == first_pass


def test_project_specific_rule_can_replace_demo_incident_identifier() -> None:
    incident_rule = MaskingRule(
        name="demo_incident_id",
        pattern=re.compile(r"INC-DEMO-\d{4}"),
        replacement="[MASCHERATO:INCIDENTE]",
    )
    masker = RegexSensitiveDataMasker(rules=[incident_rule])

    assert masker.mask("Analizzare INC-DEMO-1234") == ("Analizzare [MASCHERATO:INCIDENTE]")


@pytest.mark.parametrize(
    "text",
    [pytest.param("", id="testo-vuoto"), pytest.param("Checkout operativo.", id="testo-pulito")],
)
def test_text_without_sensitive_values_is_left_unchanged(
    default_masker: RegexSensitiveDataMasker,
    text: str,
) -> None:
    assert default_masker.mask(text) == text
