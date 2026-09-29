import pytest

from app.ingestion import SectionAwareChunker
from app.models import Document, SourceMetadata


def makePaymentRunbook(
    text: str,
    *,
    documentType: str = "markdown",
    section: str | None = None,
) -> Document:
    return Document(
        id="document-payment-runbook",
        text=text,
        metadata=SourceMetadata(
            source="runbooks/payment-unreachable.md",
            documentType=documentType,
            service="payment",
            section=section,
        ),
    )


def testSameRunbookProducesSameChunksAndIdentifiers() -> None:
    runbook = makePaymentRunbook(
        "# Errore di connessione\n" + "connection refused verso payment. " * 12
    )
    chunker = SectionAwareChunker(chunkSize=90, chunkOverlap=15)

    firstRun = chunker.chunk(runbook)
    secondRun = chunker.chunk(runbook)

    assert firstRun == secondRun
    assert [chunk.id for chunk in firstRun] == [chunk.id for chunk in secondRun]


def testMarkdownHeadingsArePreservedAsChunkSections() -> None:
    runbook = makePaymentRunbook(
        "# Sintomi\nIl checkout riceve connection refused.\n"
        "# Verifiche\nControllare lo stato del servizio payment."
    )

    chunks = SectionAwareChunker(chunkSize=200).chunk(runbook)

    assert [chunk.metadata.section for chunk in chunks] == ["Sintomi", "Verifiche"]
    assert chunks[0].text.startswith("# Sintomi")
    assert chunks[1].text.startswith("# Verifiche")


def testChunkKeepsDocumentAndServiceProvenance() -> None:
    operationalNote = makePaymentRunbook(
        "Riavviare payment solo dopo aver verificato le dipendenze.",
        documentType="text",
        section="Procedura operativa",
    )

    chunk = SectionAwareChunker(chunkSize=100).chunk(operationalNote)[0]

    assert chunk.documentId == operationalNote.id
    assert chunk.metadata.source == "runbooks/payment-unreachable.md"
    assert chunk.metadata.service == "payment"
    assert chunk.metadata.section == "Procedura operativa"


def testOverlapKeepsBoundaryTextInAdjacentChunks() -> None:
    note = makePaymentRunbook("checkoutpaymentdown", documentType="text")

    chunks = SectionAwareChunker(chunkSize=10, chunkOverlap=2).chunk(note)

    assert [chunk.text for chunk in chunks] == ["checkoutpa", "paymentdow", "own"]
    assert chunks[0].text[-2:] == chunks[1].text[:2] == "pa"
    assert chunks[1].text[-2:] == chunks[2].text[:2] == "ow"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="vuoto"),
        pytest.param("   ", id="spazi"),
        pytest.param("\n\n", id="righe-vuote"),
    ],
)
def testBlankDocumentDoesNotCreateRetrievableChunks(text: str) -> None:
    document = makePaymentRunbook(text)

    assert SectionAwareChunker(chunkSize=100).chunk(document) == []


@pytest.mark.parametrize(
    ("chunkSize", "chunkOverlap"),
    [
        pytest.param(0, 0, id="dimensione-zero"),
        pytest.param(-1, 0, id="dimensione-negativa"),
        pytest.param(10, -1, id="overlap-negativo"),
        pytest.param(10, 10, id="overlap-uguale-alla-dimensione"),
        pytest.param(10, 11, id="overlap-maggiore-della-dimensione"),
    ],
)
def testInvalidChunkBoundariesAreRejected(
    chunkSize: int,
    chunkOverlap: int,
) -> None:
    with pytest.raises(ValueError):
        SectionAwareChunker(chunkSize=chunkSize, chunkOverlap=chunkOverlap)
