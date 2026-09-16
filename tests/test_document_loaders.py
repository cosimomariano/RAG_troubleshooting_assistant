from pathlib import Path

import pytest

from app.ingestion import LocalDocumentLoader


def test_loads_markdown_with_relative_source_and_stable_id(tmp_path: Path) -> None:
    runbook = tmp_path / "runbooks" / "payment-unreachable.md"
    runbook.parent.mkdir()
    runbook.write_text("# Payment unavailable\n\nCheck the endpoint.", encoding="utf-8")
    loader = LocalDocumentLoader(tmp_path)

    first_document = loader.load_file(runbook)
    second_document = loader.load_file("runbooks/payment-unreachable.md")

    assert first_document.id == second_document.id
    assert first_document.text.startswith("# Payment unavailable")
    assert first_document.metadata.source == "runbooks/payment-unreachable.md"
    assert first_document.metadata.document_type == "markdown"


def test_loads_empty_text_document(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("", encoding="utf-8")

    document = LocalDocumentLoader(tmp_path).load_file(text_file)

    assert document.text == ""
    assert document.metadata.document_type == "text"


def test_directory_scan_is_recursive_ordered_and_ignores_unsupported_files(
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


def test_non_recursive_scan_excludes_nested_documents(tmp_path: Path) -> None:
    (tmp_path / "root.txt").write_text("Root", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.md").write_text("Child", encoding="utf-8")

    documents = LocalDocumentLoader(tmp_path, recursive=False).load()

    assert [document.metadata.source for document in documents] == ["root.txt"]


def test_directory_scan_supports_relative_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_base = Path("knowledge_base")
    knowledge_base.mkdir()
    (knowledge_base / "overview.md").write_text("Overview", encoding="utf-8")

    documents = LocalDocumentLoader(knowledge_base).load()

    assert [document.metadata.source for document in documents] == ["overview.md"]


def test_explicit_load_rejects_unsupported_extension(tmp_path: Path) -> None:
    unsupported = tmp_path / "payload.json"
    unsupported.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Estensione non supportata"):
        LocalDocumentLoader(tmp_path).load_file(unsupported)


def test_missing_knowledge_base_is_reported(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Knowledge Base non trovata"):
        LocalDocumentLoader(missing_root).load()
