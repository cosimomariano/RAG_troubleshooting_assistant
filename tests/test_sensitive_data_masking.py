import re

import pytest

from app.ingestion import MaskingRule, RegexSensitiveDataMasker, SensitiveDataMasker


@pytest.fixture
def defaultMasker() -> RegexSensitiveDataMasker:
    return RegexSensitiveDataMasker()


def testRegexMaskerCanBeUsedThroughCommonContract(
    defaultMasker: RegexSensitiveDataMasker,
) -> None:
    assert isinstance(defaultMasker, SensitiveDataMasker)


def testOnCallEmailIsRemovedFromRunbookText(
    defaultMasker: RegexSensitiveDataMasker,
) -> None:
    runbookLine = "Contattare oncall.payment@example.test in caso di errore."
    maskedLine = defaultMasker.mask(runbookLine)

    assert maskedLine == "Contattare [MASCHERATO:EMAIL] in caso di errore."


@pytest.mark.parametrize(
    ("logLine", "expectedLine"),
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
def testCredentialsAreRemovedFromTelemetryLines(
    defaultMasker: RegexSensitiveDataMasker,
    logLine: str,
    expectedLine: str,
) -> None:
    assert defaultMasker.mask(logLine) == expectedLine


def testIbanLikeValueIsNotSentToIndexing(
    defaultMasker: RegexSensitiveDataMasker,
) -> None:
    incidentNote = "IBAN usato nel test: IT60 X054 2811 1010 0000 0123 456."

    maskedNote = defaultMasker.mask(incidentNote)

    assert maskedNote == "IBAN usato nel test: [MASCHERATO:IBAN]."


def testInternalServiceAddressIsHiddenButPublicResolverIsKept(
    defaultMasker: RegexSensitiveDataMasker,
) -> None:
    logLine = "payment risponde da 10.23.4.5; il resolver configurato è 8.8.8.8."

    maskedLine = defaultMasker.mask(logLine)

    assert maskedLine == (
        "payment risponde da [MASCHERATO:IP_PRIVATO]; il resolver configurato è 8.8.8.8."
    )


def testPrivateIpv4IsHiddenWhenFollowedBySentencePeriod(
    defaultMasker: RegexSensitiveDataMasker,
) -> None:
    text = "Il servizio payment risponde da 10.23.4.5."

    assert defaultMasker.mask(text) == ("Il servizio payment risponde da [MASCHERATO:IP_PRIVATO].")


def testReprocessingMaskedTextDoesNotChangeItAgain(
    defaultMasker: RegexSensitiveDataMasker,
) -> None:
    logLine = "Email admin@example.test e password=demo-password-123."

    firstPass = defaultMasker.mask(logLine)

    assert defaultMasker.mask(logLine) == firstPass
    assert defaultMasker.mask(firstPass) == firstPass


def testProjectSpecificRuleCanReplaceDemoIncidentIdentifier() -> None:
    incidentRule = MaskingRule(
        name="demo_incident_id",
        pattern=re.compile(r"INC-DEMO-\d{4}"),
        replacement="[MASCHERATO:INCIDENTE]",
    )
    masker = RegexSensitiveDataMasker(rules=[incidentRule])

    assert masker.mask("Analizzare INC-DEMO-1234") == ("Analizzare [MASCHERATO:INCIDENTE]")


@pytest.mark.parametrize(
    "text",
    [pytest.param("", id="testo-vuoto"), pytest.param("Checkout operativo.", id="testo-pulito")],
)
def testTextWithoutSensitiveValuesIsLeftUnchanged(
    defaultMasker: RegexSensitiveDataMasker,
    text: str,
) -> None:
    assert defaultMasker.mask(text) == text
