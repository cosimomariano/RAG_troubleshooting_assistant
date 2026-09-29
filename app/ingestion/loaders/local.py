from hashlib import sha256
from pathlib import Path

from app.models import Document, SourceMetadata

SUPPORTED_EXTENSIONS = (".md", ".txt")
DOCUMENT_TYPE_BY_EXTENSION = {".md": "markdown", ".txt": "text"}


class LocalDocumentLoader:
    def __init__(self, root: str | Path, *, recursive: bool = True) -> None:
        self.root = Path(root)
        self.recursive = recursive

    def load(self) -> list[Document]:
        self.validateRootDirectory()

        documents: list[Document] = []
        for documentPath in self.findDocumentPaths():
            relativePath = documentPath.relative_to(self.root)
            documents.append(self.loadFile(relativePath))
        return documents

    def loadFile(self, path: str | Path) -> Document:
        resolvedRoot = self.root.resolve()
        resolvedPath = self.resolvePath(path, resolvedRoot)
        relativePath = self.getRelativePath(resolvedPath, resolvedRoot)

        self.validateDocumentFile(resolvedPath)

        extension = resolvedPath.suffix.lower()
        source = relativePath.as_posix()
        return Document(
            id=self.buildDocumentId(source),
            text=resolvedPath.read_text(encoding="utf-8"),
            metadata=SourceMetadata(
                source=source,
                documentType=DOCUMENT_TYPE_BY_EXTENSION[extension],
            ),
        )

    def validateRootDirectory(self) -> None:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(
                f"Knowledge Base non trovata o non correttamente censita: {self.root}"
            )

    def findDocumentPaths(self) -> list[Path]:
        pathIterator = self.root.rglob("*") if self.recursive else self.root.glob("*")
        documentPaths: list[Path] = []

        for candidatePath in pathIterator:
            if not candidatePath.is_file():
                continue
            if candidatePath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            documentPaths.append(candidatePath)

        return sorted(
            documentPaths,
            key=lambda documentPath: documentPath.relative_to(self.root).as_posix(),
        )

    @staticmethod
    def resolvePath(path: str | Path, resolvedRoot: Path) -> Path:
        candidatePath = Path(path)
        if not candidatePath.is_absolute():
            candidatePath = resolvedRoot / candidatePath
        return candidatePath.resolve()

    @staticmethod
    def getRelativePath(resolvedPath: Path, resolvedRoot: Path) -> Path:
        try:
            return resolvedPath.relative_to(resolvedRoot)
        except ValueError as error:
            raise ValueError(
                f"Il file non appartiene alla Knowledge Base: {resolvedPath}"
            ) from error

    @staticmethod
    def validateDocumentFile(documentPath: Path) -> None:
        if not documentPath.is_file():
            raise FileNotFoundError(f"Documento non trovato: {documentPath}")

        extension = documentPath.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Estensione non supportata: {extension or '<nessuna>'}")

    @staticmethod
    def buildDocumentId(source: str) -> str:
        digest = sha256(source.encode("utf-8")).hexdigest()
        return f"document-{digest}"
