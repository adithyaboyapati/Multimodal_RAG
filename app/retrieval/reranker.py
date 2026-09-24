"""
Cross-Encoder Reranking engine for precision filtering of multimodal candidate chunks.
"""
import logging
from typing import List, Optional
import torch
from sentence_transformers import CrossEncoder

from app.schemas.query import RetrievedChunk

logger = logging.getLogger("novacore.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks candidate chunks retrieved from vector search using cross-attention scoring.
    Filters out visual summaries that match conceptually but do not answer the specific query.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy initialization of the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Score (query, chunk) pairs and return top_k candidates reordered by relevance.
        """
        if not chunks:
            return []

        if len(chunks) <= 1:
            return chunks[:top_k]

        try:
            pairs = [[query, c.page_content] for c in chunks]
            scores = self.model.predict(pairs)

            # Pair chunks with new cross-encoder scores
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                # Update chunk score
                chunk.score = float(score)
                scored_chunks.append(chunk)

            # Sort descending by cross-encoder score
            scored_chunks.sort(key=lambda c: c.score, reverse=True)

            # Re-index ranks
            for rank, c in enumerate(scored_chunks, start=1):
                c.rank = rank

            return scored_chunks[:top_k]

        except Exception as exc:
            logger.warning("CrossEncoder reranking failed: %s. Falling back to retrieval rank.", exc)
            # Graceful degradation: return original chunks sorted by initial retrieval score
            sorted_original = sorted(chunks, key=lambda c: c.score, reverse=True)
            for rank, c in enumerate(sorted_original, start=1):
                c.rank = rank
            return sorted_original[:top_k]
