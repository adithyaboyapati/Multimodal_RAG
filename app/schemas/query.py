"""
Query request, intent classification, filtering, and retrieval schemas.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config.constants import ModalityType
from app.schemas.document import DocumentMetadata


class ModalityIntent(str, Enum):
    """Classified user query intent."""
    ALL = "all"
    TEXT_ONLY = "text_only"
    TABULAR = "tabular"
    VISUAL = "visual"
    CROSS_MODAL = "cross_modal"


class FilterCriteria(BaseModel):
    """Metadata filtering criteria passed to vector search."""
    pages: Optional[List[int]] = Field(default=None, description="Filter to specific page numbers")
    modalities: Optional[List[ModalityType]] = Field(default=None, description="Filter to specific modalities")
    source: Optional[str] = Field(default=None, description="Filter to specific document filename")
    document_hash: Optional[str] = Field(default=None, description="Filter to specific document hash")

    def to_pinecone_filter(self) -> Optional[Dict[str, Any]]:
        """Translate filter criteria into Pinecone metadata query filter syntax."""
        clauses = []
        if self.pages:
            clauses.append({"page": {"$in": self.pages}})
        if self.modalities:
            clauses.append({"modality": {"$in": [m.value for m in self.modalities]}})
        if self.source:
            clauses.append({"source": {"$eq": self.source}})
        if self.document_hash:
            clauses.append({"document_hash": {"$eq": self.document_hash}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}


class QueryRequest(BaseModel):
    """User input payload for multimodal search and answer generation."""
    question: str = Field(..., min_length=2, max_length=2000, description="The user query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of candidate chunks to retrieve")
    hybrid_alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weighting between dense (1.0) and sparse (0.0) search",
    )
    filters: Optional[FilterCriteria] = Field(default=None, description="Optional metadata filters")
    rerank: bool = Field(default=True, description="Apply secondary cross-modal reranking")
    return_visual_artifacts: bool = Field(default=True, description="Attach base64 images in response")
    max_visuals: int = Field(default=3, ge=0, le=5, description="Max images to evaluate in generator")


class RetrievedChunk(BaseModel):
    """An individual chunk retrieved and scored from the vector store."""
    chunk_id: str
    page_content: str
    metadata: DocumentMetadata
    score: float = Field(..., description="Cosine similarity or reranker relevance score")
    rank: int = Field(..., ge=1, description="1-indexed rank in result set")
