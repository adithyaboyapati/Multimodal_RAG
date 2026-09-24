"""
Unit tests for application settings and domain schemas.
"""
import pytest
from pydantic import ValidationError

from app.config.constants import ModalityType
from app.config.settings import Settings
from app.schemas.document import BoundingBox, DocumentChunk, DocumentMetadata
from app.schemas.query import FilterCriteria, QueryRequest
from app.schemas.response import Citation, LatencyBreakdown, RAGResponse, VisualArtifact


def test_settings_defaults():
    """Verify settings initialize with expected defaults."""
    settings = Settings(
        groq_api_key="mock_groq_key",
        pinecone_api_key="mock_pinecone_key",
    )
    assert settings.pinecone_index_name == "novacore-multimodal-rag"
    assert settings.embedding_dimension == 384
    assert settings.hybrid_alpha == 0.6
    assert settings.chunk_size == 400


def test_bounding_box_validation():
    """Verify bounding box coordinates enforce geometry constraints."""
    box = BoundingBox(x0=10.0, y0=20.0, x1=50.0, y1=80.0, page_width=612.0, page_height=792.0)
    assert box.x1 >= box.x0
    assert box.y1 >= box.y0

    with pytest.raises(ValidationError):
        # Invalid: x1 < x0
        BoundingBox(x0=50.0, y0=20.0, x1=10.0, y1=80.0)

    with pytest.raises(ValidationError):
        # Invalid: y1 < y0
        BoundingBox(x0=10.0, y0=90.0, x1=50.0, y1=80.0)


def test_document_metadata_to_pinecone():
    """Verify document metadata formats correctly for Pinecone payload."""
    meta = DocumentMetadata(
        source="report.pdf",
        document_hash="abc123hash",
        page_number=3,
        modality=ModalityType.VISUAL,
        image_path="novacore_extracted_images/p3_img1.png",
        chunk_index=1,
        bounding_box=BoundingBox(x0=10, y0=20, x1=100, y1=200),
    )
    payload = meta.to_pinecone_metadata()
    assert payload["source"] == "report.pdf"
    assert payload["page"] == 3
    assert payload["modality"] == "visual"
    assert payload["image_path"] == "novacore_extracted_images/p3_img1.png"
    assert payload["bbox_x0"] == 10.0
    assert payload["bbox_y1"] == 200.0


def test_filter_criteria_pinecone_conversion():
    """Verify metadata filter translation into Pinecone filter syntax."""
    # Single filter clause
    f1 = FilterCriteria(pages=[1, 2, 3])
    pinecone_f1 = f1.to_pinecone_filter()
    assert pinecone_f1 == {"page": {"$in": [1, 2, 3]}}

    # Multiple combined filter clauses
    f2 = FilterCriteria(
        pages=[4],
        modalities=[ModalityType.TABLE, ModalityType.VISUAL],
        source="NovaCore.pdf",
    )
    pinecone_f2 = f2.to_pinecone_filter()
    assert "$and" in pinecone_f2
    assert {"page": {"$in": [4]}} in pinecone_f2["$and"]
    assert {"modality": {"$in": ["table", "visual"]}} in pinecone_f2["$and"]
    assert {"source": {"$eq": "NovaCore.pdf"}} in pinecone_f2["$and"]


def test_rag_response_serialization():
    """Verify complete RAG response schema serializes without issues."""
    response = RAGResponse(
        question="What was Q4 revenue?",
        answer="Q4 revenue reached $42.5M [Page 3 | TABLE].",
        citations=[
            Citation(
                source="report.pdf",
                page_number=3,
                modality=ModalityType.TABLE,
                excerpt="Q4 Revenue: $42.5M",
            )
        ],
        visual_artifacts=[
            VisualArtifact(
                asset_id="asset-1",
                page_number=3,
                image_path="path/to/img.png",
            )
        ],
        retrieved_chunks_count=5,
        modality_used="text_llm",
        model="openai/gpt-oss-20b",
        latency=LatencyBreakdown(retrieval_ms=45.2, generation_ms=310.8, total_ms=356.0),
    )
    data = response.model_dump()
    assert data["question"] == "What was Q4 revenue?"
    assert len(data["citations"]) == 1
    assert data["latency"]["total_ms"] == 356.0
