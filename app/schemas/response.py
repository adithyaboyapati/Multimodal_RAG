"""
Response models for grounded RAG generation, citations, visual artifacts, and latency telemetry.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from app.config.constants import ModalityType
from app.schemas.document import BoundingBox


class Citation(BaseModel):
    """Grounded reference connecting an answer claim to source document coordinates."""
    source: str = Field(..., description="Source document filename")
    page_number: int = Field(..., ge=1, description="Page number where evidence resides")
    modality: ModalityType = Field(..., description="Modality of evidence: text, table, or visual")
    excerpt: str = Field(..., description="Short textual excerpt or description of evidence")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates if known")
    confidence_score: Optional[float] = Field(default=None, description="Retrieval or alignment score")


class VisualArtifact(BaseModel):
    """Extracted visual asset (chart, graph, diagram) retrieved to support answer."""
    asset_id: str = Field(..., description="Unique asset identifier")
    page_number: int = Field(..., ge=1, description="Source page number")
    image_path: str = Field(..., description="Storage path or URI")
    data_uri: Optional[str] = Field(default=None, description="Base64 Data URI for frontend display")
    summary: Optional[str] = Field(default=None, description="Pre-computed VLM summary of the visual")


class LatencyBreakdown(BaseModel):
    """Execution timing telemetry across RAG pipeline stages in milliseconds."""
    intent_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    """The complete, production-grade multimodal RAG response."""
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Synthesized, strictly grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="List of source citations")
    visual_artifacts: List[VisualArtifact] = Field(
        default_factory=list,
        description="Original visual images consulted during generation",
    )
    retrieved_chunks_count: int = Field(default=0, description="Total candidate chunks retrieved")
    modality_used: str = Field(
        ...,
        description="Generation pathway executed: 'text_llm' or 'multimodal_vlm'",
    )
    model: str = Field(..., description="LLM/VLM model identifier that generated the answer")
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown, description="Telemetry timings")
