"""Retrieval package — dense retrieval with top-k and MMR, plus reranking stub."""

from docintel.retrieval.reranker import NoOpReranker, Reranker
from docintel.retrieval.retriever import DenseRetriever, Retriever

__all__ = ["Retriever", "DenseRetriever", "Reranker", "NoOpReranker"]
