"""
Dense embedding implementation using OpenAI text-embedding-3-small.
Supports Matryoshka Representation Learning (MRL) dimensions parameter.
"""
from typing import List, Optional
from openai import OpenAI

from app.config.settings import Settings, get_settings
from app.embeddings.base import BaseDenseEmbedder


class OpenAIDenseEmbedder(BaseDenseEmbedder):
    """
    OpenAI dense embedding wrapper using text-embedding-3-small.
    Supports flexible dimension configuration (e.g., 384 to match existing Pinecone indexes,
    or 1536 for full dimensionality).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        settings: Optional[Settings] = None,
        client: Optional[OpenAI] = None,
    ):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model or "text-embedding-3-small"
        self._dimension = dimension or self.settings.embedding_dimension
        api_key = self.settings.openai_api_key or "placeholder_key"
        self.client = client or OpenAI(api_key=api_key)
        self._query_cache = {}

    @property
    def dimension(self) -> int:
        """Configured output dimension for embeddings."""
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string using text-embedding-3-small with in-memory caching."""
        clean_text = text.replace("\n", " ").strip()
        cache_key = (clean_text.lower(), self._dimension, self.model_name)
        if cache_key in self._query_cache:
            return list(self._query_cache[cache_key])

        kwargs = {
            "model": self.model_name,
            "input": [clean_text],
        }
        if self._dimension:
            kwargs["dimensions"] = self._dimension

        response = self.client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding

        # Cache with cap
        if len(self._query_cache) >= 1024:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = embedding

        return embedding

    def embed_documents(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Batch embed multiple document strings with dimension reduction."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i : i + batch_size]]
            kwargs = {
                "model": self.model_name,
                "input": batch,
            }
            if self._dimension:
                kwargs["dimensions"] = self._dimension

            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
