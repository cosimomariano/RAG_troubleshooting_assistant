import pytest

from app.ingestion import SectionAwareChunker
from app.models import Document, SourceMetadata


def make_payment_runbook(
    text: str,
    *,
    document_type: str = "markdown",
    section: str | None = None,
) -> Document:
    return Document(
        id="document-payment-runbook",
        text=text,
        metadata=SourceMetadata(
            source="runbooks/payment-unreachable.md",
            document_type=document_type,
            service="payment",
            section=section,
        ),
    )


def test_same_runbook_produces_same_chunks_and_identifiers() -> None:
    runbook = make_payment_runbook(
        "# Errore di connessione\n" + "connection refused verso payment. " * 12
    )
    chunker = SectionAwareChunker(chunk_size=90, chunk_overlap=15)

    first_run = chunker.chunk(runbook)
    second_run = chunker.chunk(runbook)

    assert first_run == second_run
    assert [chunk.id for chunk in first_run] == [chunk.id for chunk in second_run]


def test_markdown_headings_are_preserved_as_chunk_sections() -> None:
    runbook = make_payment_runbook(
        "# Sintomi\nIl checkout riceve connection refused.\n"
        "# Verifiche\nControllare lo stato del servizio payment."
    )

    chunks = SectionAwareChunker(chunk_size=200).chunk(runbook)

    assert [chunk.metadata.section for chunk in chunks] == ["Sintomi", "Verifiche"]
    assert chunks[0].text.startswith("# Sintomi")
    assert chunks[1].text.startswith("# Verifiche")


def test_chunk_keeps_document_and_service_provenance() -> None:
    operational_note = make_payment_runbook(
        "Riavviare payment solo dopo aver verificato le dipendenze.",
        document_type="text",
        section="Procedura operativa",
    )

    chunk = SectionAwareChunker(chunk_size=100).chunk(operational_note)[0]

    assert chunk.document_id == operational_note.id
    assert chunk.metadata.source == "runbooks/payment-unreachable.md"
    assert chunk.metadata.service == "payment"
    assert chunk.metadata.section == "Procedura operativa"


def test_overlap_keeps_boundary_text_in_adjacent_chunks() -> None:
    note = make_payment_runbook("checkoutpaymentdown", document_type="text")

    chunks = SectionAwareChunker(chunk_size=10, chunk_overlap=2).chunk(note)

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
def test_blank_document_does_not_create_retrievable_chunks(text: str) -> None:
    document = make_payment_runbook(text)

    assert SectionAwareChunker(chunk_size=100).chunk(document) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        pytest.param(0, 0, id="dimensione-zero"),
        pytest.param(-1, 0, id="dimensione-negativa"),
        pytest.param(10, -1, id="overlap-negativo"),
        pytest.param(10, 10, id="overlap-uguale-alla-dimensione"),
        pytest.param(10, 11, id="overlap-maggiore-della-dimensione"),
    ],
)
def test_invalid_chunk_boundaries_are_rejected(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        SectionAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
