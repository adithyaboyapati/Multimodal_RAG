"""
Dense embedding implementation using SentenceTransformers with hardware acceleration.
"""
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from app.config.settings import Settings, get_settings
from app.embeddings.base import BaseDenseEmbedder


class SentenceTransformerDenseEmbedder(BaseDenseEmbedder):
    """Dense embedding model wrapper with hardware acceleration (MPS/CUDA/CPU)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        default_hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        if model_name:
            self.model_name = model_name
        elif self.settings.embedding_model and not self.settings.embedding_model.startswith("text-embedding"):
            self.model_name = self.settings.embedding_model
        else:
            self.model_name = default_hf_model

        # Auto-detect optimal compute device (CPU is fastest on macOS for small MiniLM models)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dimension = self.model.get_embedding_dimension()
        else:
            self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Dimensionality of the dense vectors."""
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with L2 normalization."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch embed multiple document chunks with L2 normalization."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
