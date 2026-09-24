"""
Document domain models and schemas for the multimodal ingestion pipeline.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.config.constants import ModalityType


class BoundingBox(BaseModel):
    """Normalized spatial coordinates [x0, y0, x1, y1] for layout elements."""
    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")
    page_width: Optional[float] = Field(default=None, description="Width of the page in points")
    page_height: Optional[float] = Field(default=None, description="Height of the page in points")

    @field_validator("x1")
    @classmethod
    def validate_x(cls, v: float, info) -> float:
        if "x0" in info.data and v < info.data["x0"]:
            raise ValueError(f"x1 ({v}) must be greater than or equal to x0 ({info.data['x0']})")
        return v

    @field_validator("y1")
    @classmethod
    def validate_y(cls, v: float, info) -> float:
        if "y0" in info.data and v < info.data["y0"]:
            raise ValueError(f"y1 ({v}) must be greater than or equal to y0 ({info.data['y0']})")
        return v


class DocumentMetadata(BaseModel):
    """Structured metadata attached to every multimodal document chunk."""
    source: str = Field(..., description="Source document filename or URI")
    document_hash: str = Field(default="", description="SHA-256 hash of the source document")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    modality: ModalityType = Field(..., description="Content modality: text, table, or visual")
    table_number: Optional[int] = Field(default=None, ge=1, description="Sequential table index on page")
    image_path: Optional[str] = Field(default=None, description="Path or URI to stored visual artifact")
    parent_chunk_id: Optional[str] = Field(default=None, description="ID of parent context window")
    chunk_index: int = Field(default=0, ge=0, description="Sequential chunk index on page")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates on page")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Ingestion timestamp",
    )
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom metadata")

    def to_pinecone_metadata(self) -> Dict[str, Any]:
        """Convert metadata to flat dictionary format required by Pinecone."""
        payload = {
            "source": self.source,
            "document_hash": self.document_hash,
            "page": self.page_number,
            "modality": self.modality.value,
            "chunk_index": self.chunk_index,
        }
        if self.table_number is not None:
            payload["table_number"] = self.table_number
        if self.image_path is not None:
            payload["image_path"] = self.image_path
        if self.parent_chunk_id is not None:
            payload["parent_chunk_id"] = self.parent_chunk_id
        if self.bounding_box is not None:
            payload["bbox_x0"] = self.bounding_box.x0
            payload["bbox_y0"] = self.bounding_box.y0
            payload["bbox_x1"] = self.bounding_box.x1
            payload["bbox_y1"] = self.bounding_box.y1
        return payload


class DocumentChunk(BaseModel):
    """The atomic retrievable unit stored in the vector database."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk UUID")
    page_content: str = Field(..., min_length=1, description="The searchable text representation")
    metadata: DocumentMetadata = Field(..., description="Attached structured metadata")
    dense_vector: Optional[List[float]] = Field(default=None, description="Dense embedding vector")
    sparse_indices: Optional[List[int]] = Field(default=None, description="BM25 sparse vector term indices")
    sparse_values: Optional[List[float]] = Field(default=None, description="BM25 sparse vector term weights")


class ParentDocument(BaseModel):
    """Contextual parent container containing broader section context."""
    parent_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique parent ID")
    source: str = Field(..., description="Source document filename")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    full_content: str = Field(..., description="Full page or multi-paragraph context")
    child_chunk_ids: List[str] = Field(default_factory=list, description="List of child chunk UUIDs")


class IngestionSummary(BaseModel):
    """Summary statistics returned after completing document ingestion."""
    document_name: str
    document_hash: str
    total_pages: int
    text_chunks_count: int
    tables_count: int
    images_count: int
    total_vectors_indexed: int
    duration_seconds: float
