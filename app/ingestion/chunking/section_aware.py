import re
from hashlib import sha256

from app.models import Document, DocumentChunk, SourceMetadata

MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")


class SectionAwareChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int = 0) -> None:
        self._validate_configuration(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        document_sections = self._split_sections(document)

        for section_name, section_text in document_sections:
            text_fragments = self._split_text(section_text)
            for text_fragment in text_fragments:
                position = len(chunks)
                chunk = self._create_chunk(
                    document=document,
                    section_name=section_name,
                    position=position,
                    text=text_fragment,
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        document: Document,
        section_name: str | None,
        position: int,
        text: str,
    ) -> DocumentChunk:
        chunk_id = self._build_chunk_id(
            document_id=document.id,
            section=section_name,
            position=position,
            text=text,
        )
        metadata = self._copy_metadata(document.metadata, section_name)
        return DocumentChunk(
            id=chunk_id,
            document_id=document.id,
            text=text,
            metadata=metadata,
        )

    @staticmethod
    def _copy_metadata(
        metadata: SourceMetadata,
        section_name: str | None,
    ) -> SourceMetadata:
        return metadata.model_copy(update={"section": section_name})

    def _split_sections(self, document: Document) -> list[tuple[str | None, str]]:
        if document.metadata.document_type != "markdown":
            return [(document.metadata.section, document.text)]

        sections: list[tuple[str | None, str]] = []
        current_section_name = document.metadata.section
        current_section_lines: list[str] = []

        for line in document.text.splitlines(keepends=True):
            heading_match = MARKDOWN_HEADING_PATTERN.match(line.rstrip("\r\n"))
            if heading_match is None:
                current_section_lines.append(line)
                continue

            self._append_section(
                sections,
                current_section_name,
                current_section_lines,
            )
            current_section_name = heading_match.group(1).strip()
            current_section_lines = [line]

        self._append_section(
            sections,
            current_section_name,
            current_section_lines,
        )
        return sections

    @staticmethod
    def _append_section(
        sections: list[tuple[str | None, str]],
        section_name: str | None,
        section_lines: list[str],
    ) -> None:
        section_text = "".join(section_lines)
        if section_text.strip():
            sections.append((section_name, section_text))

    def _split_text(self, text: str) -> list[str]:
        normalized_text = text.strip()
        if not normalized_text:
            return []

        chunks: list[str] = []
        step = self._chunk_size - self._chunk_overlap
        start = 0

        while start < len(normalized_text):
            end = start + self._chunk_size
            chunk_text = normalized_text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(normalized_text):
                break
            start += step

        return chunks

    @staticmethod
    def _validate_configuration(chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("La dimensione del chunk deve essere maggiore di zero.")
        if chunk_overlap < 0:
            raise ValueError("La sovrapposizione tra chunk non può essere negativa.")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "La sovrapposizione tra chunk deve essere minore della dimensione del chunk."
            )

    @staticmethod
    def _build_chunk_id(
        document_id: str,
        section: str | None,
        position: int,
        text: str,
    ) -> str:
        payload = f"{document_id}\0{section or ''}\0{position}\0{text}"
        digest = sha256(payload.encode("utf-8")).hexdigest()
        return f"chunk-{digest}"
