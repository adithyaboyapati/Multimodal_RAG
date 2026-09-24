"""
Domain schemas and Pydantic models for the multimodal RAG platform.
"""
from app.schemas.document import (
    BoundingBox,
    DocumentChunk,
    DocumentMetadata,
    IngestionSummary,
    ParentDocument,
)
from app.schemas.query import (
    FilterCriteria,
    ModalityIntent,
    QueryRequest,
    RetrievedChunk,
)
from app.schemas.response import (
    Citation,
    LatencyBreakdown,
    RAGResponse,
    VisualArtifact,
)

__all__ = [
    "BoundingBox",
    "DocumentChunk",
    "DocumentMetadata",
    "ParentDocument",
    "IngestionSummary",
    "ModalityIntent",
    "FilterCriteria",
    "QueryRequest",
    "RetrievedChunk",
    "Citation",
    "VisualArtifact",
    "LatencyBreakdown",
    "RAGResponse",
]
