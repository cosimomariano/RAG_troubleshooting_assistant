from hashlib import sha256
from pathlib import Path

from app.models import Document, SourceMetadata

SUPPORTED_EXTENSIONS = (".md", ".txt")
DOCUMENT_TYPE_BY_EXTENSION = {".md": "markdown", ".txt": "text"}


class LocalDocumentLoader:
    def __init__(self, root: str | Path, *, recursive: bool = True) -> None:
        self._root = Path(root)
        self._recursive = recursive

    @property
    def root(self) -> Path:
        return self._root

    @property
    def recursive(self) -> bool:
        return self._recursive

    def load(self) -> list[Document]:
        self._validate_root_directory()

        documents: list[Document] = []
        for document_path in self._find_document_paths():
            relative_path = document_path.relative_to(self._root)
            documents.append(self.load_file(relative_path))
        return documents

    def load_file(self, path: str | Path) -> Document:
        resolved_root = self._root.resolve()
        resolved_path = self._resolve_path(path, resolved_root)
        relative_path = self._get_relative_path(resolved_path, resolved_root)

        self._validate_document_file(resolved_path)

        extension = resolved_path.suffix.lower()
        source = relative_path.as_posix()
        return Document(
            id=self._build_document_id(source),
            text=resolved_path.read_text(encoding="utf-8"),
            metadata=SourceMetadata(
                source=source,
                document_type=DOCUMENT_TYPE_BY_EXTENSION[extension],
            ),
        )

    def _validate_root_directory(self) -> None:
        if not self._root.exists() or not self._root.is_dir():
            raise FileNotFoundError(
                f"Knowledge Base non trovata o non correttamente censita: {self._root}"
            )

    def _find_document_paths(self) -> list[Path]:
        path_iterator = self._root.rglob("*") if self._recursive else self._root.glob("*")
        document_paths: list[Path] = []

        for candidate_path in path_iterator:
            if not candidate_path.is_file():
                continue
            if candidate_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            document_paths.append(candidate_path)

        return sorted(
            document_paths,
            key=lambda document_path: document_path.relative_to(self._root).as_posix(),
        )

    @staticmethod
    def _resolve_path(path: str | Path, resolved_root: Path) -> Path:
        candidate_path = Path(path)
        if not candidate_path.is_absolute():
            candidate_path = resolved_root / candidate_path
        return candidate_path.resolve()

    @staticmethod
    def _get_relative_path(resolved_path: Path, resolved_root: Path) -> Path:
        try:
            return resolved_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"Il file non appartiene alla Knowledge Base: {resolved_path}"
            ) from error

    @staticmethod
    def _validate_document_file(document_path: Path) -> None:
        if not document_path.is_file():
            raise FileNotFoundError(f"Documento non trovato: {document_path}")

        extension = document_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Estensione non supportata: {extension or '<nessuna>'}")

    @staticmethod
    def _build_document_id(source: str) -> str:
        digest = sha256(source.encode("utf-8")).hexdigest()
        return f"document-{digest}"
