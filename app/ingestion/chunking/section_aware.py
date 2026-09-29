import re
from hashlib import sha256

from app.models import Document, DocumentChunk, SourceMetadata

MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")


class SectionAwareChunker:
    def __init__(self, chunkSize: int, chunkOverlap: int = 0) -> None:
        self.validateConfiguration(chunkSize, chunkOverlap)
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        documentSections = self.splitSections(document)

        for sectionName, sectionText in documentSections:
            textFragments = self.splitText(sectionText)
            for textFragment in textFragments:
                position = len(chunks)
                chunk = self.createChunk(
                    document=document,
                    sectionName=sectionName,
                    position=position,
                    text=textFragment,
                )
                chunks.append(chunk)

        return chunks

    def createChunk(
        self,
        document: Document,
        sectionName: str | None,
        position: int,
        text: str,
    ) -> DocumentChunk:
        chunkId = self.buildChunkId(
            documentId=document.id,
            section=sectionName,
            position=position,
            text=text,
        )
        metadata = self.copyMetadata(document.metadata, sectionName)
        return DocumentChunk(
            id=chunkId,
            documentId=document.id,
            text=text,
            metadata=metadata,
        )

    @staticmethod
    def copyMetadata(
        metadata: SourceMetadata,
        sectionName: str | None,
    ) -> SourceMetadata:
        return metadata.model_copy(update={"section": sectionName})

    def splitSections(self, document: Document) -> list[tuple[str | None, str]]:
        if document.metadata.documentType != "markdown":
            return [(document.metadata.section, document.text)]

        sections: list[tuple[str | None, str]] = []
        currentSectionName = document.metadata.section
        currentSectionLines: list[str] = []

        for line in document.text.splitlines(keepends=True):
            headingMatch = MARKDOWN_HEADING_PATTERN.match(line.rstrip("\r\n"))
            if headingMatch is None:
                currentSectionLines.append(line)
                continue

            self.appendSection(
                sections,
                currentSectionName,
                currentSectionLines,
            )
            currentSectionName = headingMatch.group(1).strip()
            currentSectionLines = [line]

        self.appendSection(
            sections,
            currentSectionName,
            currentSectionLines,
        )
        return sections

    @staticmethod
    def appendSection(
        sections: list[tuple[str | None, str]],
        sectionName: str | None,
        sectionLines: list[str],
    ) -> None:
        sectionText = "".join(sectionLines)
        if sectionText.strip():
            sections.append((sectionName, sectionText))

    def splitText(self, text: str) -> list[str]:
        normalizedText = text.strip()
        if not normalizedText:
            return []

        chunks: list[str] = []
        step = self.chunkSize - self.chunkOverlap
        start = 0

        while start < len(normalizedText):
            end = start + self.chunkSize
            chunkText = normalizedText[start:end].strip()
            if chunkText:
                chunks.append(chunkText)
            if end >= len(normalizedText):
                break
            start += step

        return chunks

    @staticmethod
    def validateConfiguration(chunkSize: int, chunkOverlap: int) -> None:
        if chunkSize <= 0:
            raise ValueError("La dimensione del chunk deve essere maggiore di zero.")
        if chunkOverlap < 0:
            raise ValueError("La sovrapposizione tra chunk non può essere negativa.")
        if chunkOverlap >= chunkSize:
            raise ValueError(
                "La sovrapposizione tra chunk deve essere minore della dimensione del chunk."
            )

    @staticmethod
    def buildChunkId(
        documentId: str,
        section: str | None,
        position: int,
        text: str,
    ) -> str:
        payload = f"{documentId}\0{section or ''}\0{position}\0{text}"
        digest = sha256(payload.encode("utf-8")).hexdigest()
        return f"chunk-{digest}"
