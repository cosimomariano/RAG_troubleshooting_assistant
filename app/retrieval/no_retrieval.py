from app.models import RetrievalResult

class NoRetrievalRetriever:
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return []