from pathlib import Path

import pytest

from app.ingestion import LocalDocumentLoader


def testLoadsMarkdownWithRelativeSourceAndStableId(tmp_path: Path) -> None:
    runbook = tmp_path / "runbooks" / "payment-unreachable.md"
    runbook.parent.mkdir()
    runbook.write_text("# Payment unavailable\n\nCheck the endpoint.", encoding="utf-8")
    loader = LocalDocumentLoader(tmp_path)

    firstDocument = loader.loadFile(runbook)
    secondDocument = loader.loadFile("runbooks/payment-unreachable.md")

    assert firstDocument.id == secondDocument.id
    assert firstDocument.text.startswith("# Payment unavailable")
    assert firstDocument.metadata.source == "runbooks/payment-unreachable.md"
    assert firstDocument.metadata.documentType == "markdown"


def testLoadsEmptyTextDocument(tmp_path: Path) -> None:
    textFile = tmp_path / "notes.txt"
    textFile.write_text("", encoding="utf-8")

    document = LocalDocumentLoader(tmp_path).loadFile(textFile)

    assert document.text == ""
    assert document.metadata.documentType == "text"


def testDirectoryScanIsRecursiveOrderedAndIgnoresUnsupportedFiles(
    tmp_path: Path,
) -> None:
    (tmp_path / "z-last.txt").write_text("Last", encoding="utf-8")
    nested = tmp_path / "architecture"
    nested.mkdir()
    (nested / "overview.md").write_text("Overview", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    documents = LocalDocumentLoader(tmp_path).load()

    assert [document.metadata.source for document in documents] == [
        "architecture/overview.md",
        "z-last.txt",
    ]


def testNonRecursiveScanExcludesNestedDocuments(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("Root", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.md").write_text("Child", encoding="utf-8")

    documents = LocalDocumentLoader(tmp_path, recursive=False).load()

    assert [document.metadata.source for document in documents] == ["root.txt"]


def testDirectoryScanSupportsRelativeRoot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    knowledgeBase = Path("knowledge_base")
    knowledgeBase.mkdir()
    (knowledgeBase / "overview.md").write_text("Overview", encoding="utf-8")

    documents = LocalDocumentLoader(knowledgeBase).load()

    assert [document.metadata.source for document in documents] == ["overview.md"]


def testExplicitLoadRejectsUnsupportedExtension(tmp_path: Path) -> None:
    unsupported = tmp_path / "payload.json"
    unsupported.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Estensione non supportata"):
        LocalDocumentLoader(tmp_path).loadFile(unsupported)


def testMissingKnowledgeBaseIsReported(tmp_path: Path) -> None:
    missingRoot = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Knowledge Base non trovata"):
        LocalDocumentLoader(missingRoot).load()
