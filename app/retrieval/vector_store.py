"""
Pinecone Serverless Hybrid Vector Store manager.
Integrates dense vectors, BM25 sparse vectors, alpha convex fusion, and metadata-scoped deletion.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger("novacore.retrieval.vector_store")

from app.config.constants import ModalityType
from app.config.settings import Settings, get_settings
from app.embeddings.base import BaseDenseEmbedder, BaseSparseEmbedder
from app.embeddings.dense import SentenceTransformerDenseEmbedder
from app.embeddings.sparse import BM25SparseEmbedder
from app.schemas.document import BoundingBox, DocumentChunk, DocumentMetadata
from app.schemas.query import FilterCriteria, RetrievedChunk


class PineconeHybridVectorStore:
    """
    Manages Pinecone Serverless hybrid index operations:
    - Dual Dense + Sparse upsert
    - Convex alpha search weighting
    - Metadata filtering
    - Non-destructive document-hash deletion
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        dense_embedder: Optional[BaseDenseEmbedder] = None,
        sparse_embedder: Optional[BaseSparseEmbedder] = None,
        client: Optional[Pinecone] = None,
    ):
        self.settings = settings or get_settings()
        if dense_embedder is not None:
            self.dense_embedder = dense_embedder
        elif (
            self.settings.embedding_provider == "openai"
            and self.settings.openai_api_key
            and not self.settings.openai_api_key.startswith("your_")
        ):
            from app.embeddings.openai_embedder import OpenAIDenseEmbedder
            self.dense_embedder = OpenAIDenseEmbedder(
                model_name=self.settings.embedding_model or "text-embedding-3-small",
                dimension=self.settings.embedding_dimension,
                settings=self.settings,
            )
        else:
            self.dense_embedder = SentenceTransformerDenseEmbedder(settings=self.settings)

        self.sparse_embedder = sparse_embedder or BM25SparseEmbedder()
        self.client = client or Pinecone(api_key=self.settings.pinecone_api_key)
        self.index_name = self.settings.pinecone_index_name
        self._index = None

    @property
    def index(self):
        """Lazy connection to Pinecone index."""
        if self._index is None:
            self._ensure_index()
            self._index = self.client.Index(self.index_name)
        return self._index

    def _ensure_index(self) -> None:
        """Verify index exists with correct dimensions; create if missing."""
        if getattr(self, "_index_initialized", False):
            return
        try:
            desc = self.client.describe_index(self.index_name)
            m = getattr(desc, "metric", None)
            if isinstance(m, str):
                self._metric = m.lower()
            elif isinstance(desc, dict) and "metric" in desc and isinstance(desc["metric"], str):
                self._metric = desc["metric"].lower()
            else:
                self._metric = "dotproduct"
            self._index_initialized = True
        except Exception:
            if not self.client.has_index(self.index_name):
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dense_embedder.dimension,
                    metric="dotproduct",  # dotproduct enables hybrid sparse + dense search
                    spec=ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region,
                    ),
                )
                while not self.client.describe_index(self.index_name).status["ready"]:
                    time.sleep(1.0)
            self._metric = "dotproduct"
            self._index_initialized = True

    @property
    def metric(self) -> str:
        """Cached index distance metric (e.g. 'dotproduct' or 'cosine')."""
        if getattr(self, "_metric", None) is None:
            self._ensure_index()
        return getattr(self, "_metric", "dotproduct") or "dotproduct"

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        namespace: Optional[str] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Embed and upsert multimodal chunks into Pinecone hybrid vector index.
        """
        if not chunks:
            return 0

        target_namespace = namespace or self.settings.pinecone_namespace
        texts = [c.page_content for c in chunks]

        # 1. Generate dense embeddings
        dense_vectors = self.dense_embedder.embed_documents(texts)

        # 2. Generate sparse lexical vectors (if supported by index metric)
        sparse_vectors = (
            self.sparse_embedder.encode_documents(texts)
            if self.metric == "dotproduct"
            else [{} for _ in texts]
        )

        # 3. Format Pinecone vector records
        records = []
        for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vectors)):
            meta_dict = chunk.metadata.to_pinecone_metadata()
            # Retain text excerpt in metadata for direct reconstruction (capped to 2000 chars)
            meta_dict["text"] = chunk.page_content[:2000]

            record: Dict[str, Any] = {
                "id": chunk.chunk_id,
                "values": dense_vec,
                "metadata": meta_dict,
            }
            if self.metric == "dotproduct" and i < len(sparse_vectors):
                sparse_dict = sparse_vectors[i]
                if sparse_dict.get("indices"):
                    record["sparse_values"] = {
                        "indices": sparse_dict["indices"],
                        "values": sparse_dict["values"],
                    }

            records.append(record)

        # 4. Upsert in batches
        total_upserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=target_namespace)
            total_upserted += len(batch)

        return total_upserted

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        filters: Optional[FilterCriteria] = None,
        namespace: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute alpha-weighted hybrid search across dense vectors and BM25 sparse vectors.
        Score = alpha * Dense_Score + (1 - alpha) * Sparse_Score
        Falls back seamlessly to dense cosine search if index is configured with cosine metric.
        """
        k = top_k or self.settings.top_k
        weight_alpha = alpha if alpha is not None else self.settings.hybrid_alpha
        target_namespace = namespace or self.settings.pinecone_namespace

        # 1. Dense query vector
        dense_query = self.dense_embedder.embed_query(query)
        is_hybrid = (self.metric == "dotproduct")

        if is_hybrid:
            scaled_dense = [v * weight_alpha for v in dense_query]
            sparse_query = self.sparse_embedder.encode_query(query)
            scaled_sparse_values = [v * (1.0 - weight_alpha) for v in sparse_query["values"]]

            query_kwargs: Dict[str, Any] = {
                "vector": scaled_dense,
                "top_k": k,
                "namespace": target_namespace,
                "include_metadata": True,
            }

            if sparse_query["indices"] and (1.0 - weight_alpha) > 0.0:
                query_kwargs["sparse_vector"] = {
                    "indices": sparse_query["indices"],
                    "values": scaled_sparse_values,
                }
        else:
            # Cosine index fallback: sparse vectors not supported by Pinecone serverless on cosine
            query_kwargs = {
                "vector": dense_query,
                "top_k": k,
                "namespace": target_namespace,
                "include_metadata": True,
            }

        # 3. Apply Pinecone metadata filters
        if filters:
            pinecone_filter = filters.to_pinecone_filter()
            if pinecone_filter:
                query_kwargs["filter"] = pinecone_filter

        # 4. Execute vector search
        response = self.index.query(**query_kwargs)

        # 5. Hydrate matches into typed RetrievedChunk instances
        retrieved: List[RetrievedChunk] = []
        for rank, match in enumerate(response.get("matches", []), start=1):
            meta = match.get("metadata", {})
            page_content = meta.get("text", "")

            bbox = None
            if "bbox_x0" in meta and meta.get("bbox_x0") is not None:
                try:
                    bbox = BoundingBox(
                        x0=float(meta["bbox_x0"]),
                        y0=float(meta["bbox_y0"]),
                        x1=float(meta["bbox_x1"]),
                        y1=float(meta["bbox_y1"]),
                    )
                except Exception:
                    bbox = None

            raw_mod = meta.get("modality", "text")
            try:
                modality = ModalityType(raw_mod)
            except (ValueError, TypeError):
                modality = ModalityType.TEXT

            page_num = 1
            if "page" in meta:
                try:
                    page_num = int(meta["page"])
                except (ValueError, TypeError):
                    page_num = 1
            elif "page_number" in meta:
                try:
                    page_num = int(meta["page_number"])
                except (ValueError, TypeError):
                    page_num = 1

            chunk_idx = 0
            if "chunk_index" in meta:
                try:
                    chunk_idx = int(meta["chunk_index"])
                except (ValueError, TypeError):
                    chunk_idx = 0

            tbl_num = None
            if "table_number" in meta and meta["table_number"] is not None:
                try:
                    tbl_num = int(meta["table_number"])
                except (ValueError, TypeError):
                    tbl_num = None

            doc_meta = DocumentMetadata(
                source=meta.get("source", "unknown"),
                document_hash=meta.get("document_hash", ""),
                page_number=page_num,
                modality=modality,
                table_number=tbl_num,
                image_path=meta.get("image_path"),
                parent_chunk_id=meta.get("parent_chunk_id"),
                chunk_index=chunk_idx,
                bounding_box=bbox,
            )

            retrieved.append(
                RetrievedChunk(
                    chunk_id=match["id"],
                    page_content=page_content,
                    metadata=doc_meta,
                    score=float(match.get("score", 0.0)),
                    rank=rank,
                )
            )

        return retrieved

    def delete_document(self, document_hash: str, namespace: Optional[str] = None) -> None:
        """
        Idempotent deletion: Removes all vectors belonging to a document hash
        without destroying unrelated documents in the namespace.
        """
        target_namespace = namespace or self.settings.pinecone_namespace
        try:
            self.index.delete(
                filter={"document_hash": {"$eq": document_hash}},
                namespace=target_namespace,
            )
            logger.info("Deleted existing vectors for document hash %s from namespace '%s'", document_hash, target_namespace)
        except Exception as exc:
            logger.warning("Pinecone deletion by hash '%s' bypassed or failed: %s", document_hash, exc)
