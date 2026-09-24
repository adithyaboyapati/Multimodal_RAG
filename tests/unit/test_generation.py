"""
Unit tests for token-budgeted context assembly, multimodal VLM/LLM generation, and citations.
"""
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import pytest

from app.config.constants import ModalityType
from app.config.settings import Settings
from app.generation.context_builder import ContextBuilder
from app.generation.multimodal_generator import MultimodalGenerator
from app.schemas.document import BoundingBox, DocumentMetadata
from app.schemas.query import RetrievedChunk
from app.schemas.response import LatencyBreakdown
from app.storage.asset_store import AssetStore


@pytest.fixture
def mock_asset_store(tmp_path):
    custom_settings = Settings(
        groq_api_key="mock_key",
        pinecone_api_key="mock_key",
        storage_dir=tmp_path / "assets",
    )
    return AssetStore(settings=custom_settings)


def test_context_builder_formatting_and_visual_serialization(mock_asset_store, tmp_path):
    """Verify context assembler formats headers, respects budgets, and packs images."""
    # Create valid image file
    img_path = tmp_path / "chart.png"
    img = Image.new("RGB", (100, 100), color="green")
    img.save(img_path, format="PNG")

    builder = ContextBuilder(asset_store=mock_asset_store)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            page_content="Total revenue reached $132.0M in FY2026.",
            metadata=DocumentMetadata(
                source="report.pdf",
                page_number=2,
                modality=ModalityType.TEXT,
            ),
            score=0.92,
            rank=1,
        ),
        RetrievedChunk(
            chunk_id="c2",
            page_content="Revenue trend chart showing quarterly growth from Q1 to Q4.",
            metadata=DocumentMetadata(
                source="report.pdf",
                page_number=3,
                modality=ModalityType.VISUAL,
                image_path=str(img_path),
                bounding_box=BoundingBox(x0=50, y0=50, x1=400, y1=300),
            ),
            score=0.89,
            rank=2,
        ),
    ]

    assembled = builder.build_context(chunks=chunks, max_visuals=2)

    assert "[Page 2 | TEXT | Source: report.pdf]" in assembled.context_text
    assert "[Page 3 | VISUAL | Source: report.pdf]" in assembled.context_text
    assert len(assembled.candidate_citations) == 2
    assert assembled.candidate_citations[0].page_number == 2
    assert assembled.candidate_citations[1].page_number == 3

    assert len(assembled.visual_artifacts) == 1
    artifact = assembled.visual_artifacts[0]
    assert artifact.page_number == 3
    assert artifact.data_uri.startswith("data:image/jpeg;base64,")


def test_multimodal_generator_text_and_vision_paths(tmp_path):
    """Verify generator properly routes between text LLM and multimodal VLM."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "FY2026 revenue was $132.0M [Page 2 | TEXT]."
    mock_response = MagicMock(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response

    settings = Settings(
        groq_api_key="mock_key",
        pinecone_api_key="mock_key",
        text_model="openai/gpt-oss-20b",
        vision_model="qwen/qwen3.8-27b",
    )

    generator = MultimodalGenerator(
        settings=settings,
        client=mock_client,
    )

    # 1. Text-Only Query Route
    text_chunks = [
        RetrievedChunk(
            chunk_id="c1",
            page_content="Text context",
            metadata=DocumentMetadata(source="doc.pdf", page_number=2, modality=ModalityType.TEXT),
            score=0.9,
            rank=1,
        )
    ]
    resp_text = generator.generate(
        question="What was revenue?",
        retrieved_chunks=text_chunks,
        latency=LatencyBreakdown(retrieval_ms=25.0),
    )
    assert resp_text.modality_used == "text_llm"
    assert resp_text.model == "openai/gpt-oss-20b"
    assert resp_text.latency.generation_ms > 0
    assert resp_text.latency.total_ms > 25.0
    # Check text model called
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "openai/gpt-oss-20b"

    # 2. Multimodal Vision Route
    img_path = tmp_path / "chart_v.png"
    Image.new("RGB", (100, 100), color="red").save(img_path, format="PNG")

    visual_chunks = [
        RetrievedChunk(
            chunk_id="c2",
            page_content="Visual context chart",
            metadata=DocumentMetadata(
                source="doc.pdf",
                page_number=3,
                modality=ModalityType.VISUAL,
                image_path=str(img_path),
            ),
            score=0.95,
            rank=1,
        )
    ]
    resp_vision = generator.generate(
        question="What does the chart show?",
        retrieved_chunks=visual_chunks,
    )
    assert resp_vision.modality_used == "multimodal_vlm"
    assert resp_vision.model == "qwen/qwen3.8-27b"
    assert len(resp_vision.visual_artifacts) == 1
    call_args_v = mock_client.chat.completions.create.call_args[1]
    assert call_args_v["model"] == "qwen/qwen3.8-27b"
