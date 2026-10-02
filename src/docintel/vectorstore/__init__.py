"""Vector store package — persist and query chunk embeddings."""

from docintel.vectorstore.base import RetrievedChunk, VectorStore
from docintel.vectorstore.chroma_store import ChromaStore
from docintel.vectorstore.registry import get_store

__all__ = ["VectorStore", "RetrievedChunk", "ChromaStore", "get_store"]
