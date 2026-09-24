"""
Abstract interfaces for dense semantic embeddings and sparse lexical embeddings.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDenseEmbedder(ABC):
    """Abstract interface for dense neural embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a dense vector."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document strings into dense vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of the model."""
        pass


class BaseSparseEmbedder(ABC):
    """Abstract interface for sparse lexical (BM25/SPLADE) encoders."""

    @abstractmethod
    def encode_query(self, text: str) -> Dict[str, Any]:
        """Encode a query string into a sparse vector dict {'indices': [...], 'values': [...]}."""
        pass

    @abstractmethod
    def encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encode a batch of document strings into sparse vector dicts."""
        pass
