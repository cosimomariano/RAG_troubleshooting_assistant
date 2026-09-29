from pathlib import Path
from typing import Any

import yaml


class YamlObjectReader:
    def read(
        self,
        configurationPath: str | Path,
        configurationType: str,
    ) -> dict[str, Any]:
        path = Path(configurationPath)
        if not path.is_file():
            raise FileNotFoundError(f"Configurazione {configurationType} non trovata: {path}")

        try:
            serializedConfiguration = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(
                f"La configurazione {configurationType} contiene YAML non valido."
            ) from error

        if not isinstance(serializedConfiguration, dict):
            raise ValueError(f"La configurazione {configurationType} deve essere un oggetto YAML.")
        return serializedConfiguration
