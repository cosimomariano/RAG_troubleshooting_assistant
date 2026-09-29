from pathlib import Path

import pytest

from app.config import YamlObjectReader


def testYamlObjectReaderReturnsTheRootObject(tmp_path: Path) -> None:
    configurationPath = tmp_path / "configuration.yaml"
    configurationPath.write_text(
        "retrieval:\n  mode: hybrid\n",
        encoding="utf-8",
    )

    configuration = YamlObjectReader().read(configurationPath, "di test")

    assert configuration == {"retrieval": {"mode": "hybrid"}}


@pytest.mark.parametrize(
    ("content", "expectedMessage"),
    [
        pytest.param("- dense\n- sparse\n", "oggetto YAML", id="radice-non-oggetto"),
        pytest.param("retrieval: [\n", "YAML non valido", id="yaml-non-valido"),
    ],
)
def testYamlObjectReaderRejectsInvalidDocuments(
    tmp_path: Path,
    content: str,
    expectedMessage: str,
) -> None:
    configurationPath = tmp_path / "invalid.yaml"
    configurationPath.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=expectedMessage):
        YamlObjectReader().read(configurationPath, "di test")


def testYamlObjectReaderRejectsMissingFiles(tmp_path: Path) -> None:
    configurationPath = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configurazione di test non trovata"):
        YamlObjectReader().read(configurationPath, "di test")
