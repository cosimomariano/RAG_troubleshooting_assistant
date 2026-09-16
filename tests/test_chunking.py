import pytest
from app.ingestion import SectionAwareChunker
from app.models import Document, SourceMetadata

def build_document( text: str, *, document_type: str = "markdown", section: str | None = None) -> Document:
    return Document(
        id="document-example",
        text=text,
        metadata=SourceMetadata( source="runbooks/payment.md", document_type=document_type, service="payment", section=section))

def test_chunking_is_deterministic() -> None:
    document = build_document("# Failure\n" + "connection refused " * 20)
    chunker = SectionAwareChunker(chunk_size=80, chunk_overlap=10)

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert first_run == second_run
    assert [chunk.id for chunk in first_run] == [chunk.id for chunk in second_run]

def test_markdown_headings_define_separate_sections() -> None:
    document = build_document("# Symptoms\nTimeout observed.\n# Checks\nVerify payment service.")

    chunks = SectionAwareChunker(chunk_size=200).chunk(document)

    assert [chunk.metadata.section for chunk in chunks] == ["Symptoms", "Checks"]
    assert chunks[0].text.startswith("# Symptoms")
    assert chunks[1].text.startswith("# Checks")


def test_chunk_preserves_source_and_document_provenance() -> None:
    document = build_document("Operational note", document_type="text", section="Operations")

    chunk = SectionAwareChunker(chunk_size=100).chunk(document)[0]

    assert chunk.document_id == document.id
    assert chunk.metadata.source == document.metadata.source
    assert chunk.metadata.service == "payment"
    assert chunk.metadata.section == "Operations"


def test_character_overlap_is_applied_within_section() -> None:
    document = build_document("abcdefghijklmnopqrst", document_type="text")

    chunks = SectionAwareChunker(chunk_size=10, chunk_overlap=2).chunk(document)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrst"]
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].text[-2:] == chunks[2].text[:2]


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_document_produces_no_chunks(text: str) -> None:
    document = build_document(text)

    assert SectionAwareChunker(chunk_size=100).chunk(document) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_configuration_is_rejected( chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        SectionAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)