"""Interfacce e implementazioni per il recupero della conoscenza."""

from app.retrieval.base import Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import RankFusion, ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.no_retrieval import NoRetrievalRetriever
from app.retrieval.selection import RetrievalMode, RetrieverSelector
from app.retrieval.sparse import SparseRetriever

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "NoRetrievalRetriever",
    "RankFusion",
    "ReciprocalRankFusion",
    "RetrievalMode",
    "Retriever",
    "RetrieverSelector",
    "SparseRetriever",
]
