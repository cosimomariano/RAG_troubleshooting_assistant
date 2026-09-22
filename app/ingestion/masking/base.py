from abc import ABC, abstractmethod


class SensitiveDataMasker(ABC):
    @abstractmethod
    def mask(self, text: str) -> str:
        raise NotImplementedError
