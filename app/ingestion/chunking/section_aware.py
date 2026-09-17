import re
from hashlib import sha256
from app.models import Document, DocumentChunk

MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")

class SectionAwareChunker:
    #Chunking dei documetni
    def __init__(self, chunk_size: int, chunk_overlap: int = 0) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Errore in fase di chunking")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for section, section_text in self._split_sections(document):
            for chunk_text in self._split_text(section_text):
                position = len(chunks)
                identifier = self._build_chunk_id( document_id=document.id, section=section, position=position, text=chunk_text)
                metadata = document.metadata.model_copy(update={"section": section})
                chunks.append(
                    DocumentChunk( id=identifier, document_id=document.id, text=chunk_text, metadata=metadata)
                )

        return chunks

    def _split_sections(self, document: Document) -> list[tuple[str | None, str]]:
        if document.metadata.document_type != "markdown":
            return [(document.metadata.section, document.text)]

        sections: list[tuple[str | None, str]] = []
        current_section = document.metadata.section
        current_lines: list[str] = []

        for line in document.text.splitlines(keepends=True):
            heading = MARKDOWN_HEADING.match(line.rstrip("\r\n"))
            if heading:
                if "".join(current_lines).strip():
                    sections.append((current_section, "".join(current_lines)))
                current_section = heading.group(1).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if "".join(current_lines).strip():
            sections.append((current_section, "".join(current_lines)))

        return sections

    def _split_text(self, text: str) -> list[str]:
        normalized_text = text.strip()
        if not normalized_text:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        for start in range(0, len(normalized_text), step):
            chunk_text = normalized_text[start : start + self.chunk_size].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if start + self.chunk_size >= len(normalized_text):
                break

        return chunks

    @staticmethod
    def _build_chunk_id( document_id: str, section: str | None, position: int, text: str) -> str:
        payload = f"{document_id}\0{section or ''}\0{position}\0{text}"
        return f"chunk-{sha256(payload.encode('utf-8')).hexdigest()}"