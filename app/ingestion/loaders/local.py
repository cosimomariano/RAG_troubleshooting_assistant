"""Caricamento deterministico di documenti Markdown e testuali locali."""

from hashlib import sha256
from pathlib import Path

from app.models import Document, SourceMetadata

SUPPORTED_EXTENSIONS = (".md", ".txt")
DOCUMENT_TYPE_BY_EXTENSION = {
    ".md": "markdown",
    ".txt": "text",
}


class LocalDocumentLoader:
    """Carica documenti UTF-8 contenuti in una Knowledge Base locale."""

    def __init__(self, root: str | Path, *, recursive: bool = True) -> None:
        self.root = Path(root)
        self.recursive = recursive

    def load(self) -> list[Document]:
        """Carica tutti i documenti supportati in ordine deterministico."""
        if not self.root.exists():
            raise FileNotFoundError(f"Knowledge Base non trovata: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Il percorso non è una directory: {self.root}")

        iterator = self.root.rglob("*") if self.recursive else self.root.glob("*")
        paths = sorted(
            (
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.relative_to(self.root).as_posix(),
        )
        return [self.load_file(path.relative_to(self.root)) for path in paths]

    def load_file(self, path: str | Path) -> Document:
        """Carica un singolo file supportato appartenente alla Knowledge Base."""
        root = self.root.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()

        try:
            relative_path = candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Il file non appartiene alla Knowledge Base: {candidate}") from error

        if not candidate.is_file():
            raise FileNotFoundError(f"Documento non trovato: {candidate}")

        extension = candidate.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Estensione non supportata: {extension or '<nessuna>'}")

        source = relative_path.as_posix()
        document_id = f"document-{sha256(source.encode('utf-8')).hexdigest()}"

        return Document(
            id=document_id,
            text=candidate.read_text(encoding="utf-8"),
            metadata=SourceMetadata(
                source=source,
                document_type=DOCUMENT_TYPE_BY_EXTENSION[extension],
            ),
        )
