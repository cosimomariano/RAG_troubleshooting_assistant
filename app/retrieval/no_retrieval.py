from app.models import RetrievalResult


class NoRetrievalRetriever:
    # In questa configurazione non è previsto il recupero delle fonti (baseline)
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return []
