"""
Unit tests for hierarchical chunking, parent-child context linkages, and visual summarization.
"""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.chunking.hierarchical import HierarchicalChunker
from app.chunking.visual_summarizer import VisualSummarizer
from app.config.constants import ModalityType
from app.config.settings import Settings
from app.parsing.base import (
    ExtractedTableBlock,
    ExtractedTextBlock,
    ExtractedVisualBlock,
    ParsedDocumentBundle,
    ParsedPage,
)
from app.schemas.document import BoundingBox
from app.storage.asset_store import AssetStore


@pytest.fixture
def mock_visual_summarizer(tmp_path):
    """Visual summarizer with mocked client and local asset store."""
    custom_settings = Settings(
        groq_api_key="mock_key",
        pinecone_api_key="mock_key",
        storage_dir=tmp_path / "assets",
    )
    asset_store = AssetStore(settings=custom_settings)
    mock_client = MagicMock()
    # Mock VLM response
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked visual summary: Line chart showing Q1 to Q4 growth."
    mock_response = MagicMock(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response

    summarizer = VisualSummarizer(
        settings=custom_settings,
        asset_store=asset_store,
        client=mock_client,
    )
    return summarizer, asset_store


def test_text_splitting_logic():
    """Verify recursive text splitter respects token boundaries and overlap."""
    chunker = HierarchicalChunker()
    long_text = "Sentence one. " * 50  # ~700 chars
    splits = chunker._split_text(long_text, max_chars=200, overlap_chars=40)
    assert len(splits) > 1
    for s in splits:
        assert len(s) <= 200


def test_visual_summarizer_caching(mock_visual_summarizer, tmp_path):
    """Verify that cached summaries avoid duplicate VLM network calls."""
    summarizer, asset_store = mock_visual_summarizer

    # Create dummy image file and pre-seeded cache file
    img_path = tmp_path / "test_chart.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")
    cache_path = tmp_path / "test_chart.summary.txt"
    cache_path.write_text("Pre-cached summary from disk", encoding="utf-8")

    visual = ExtractedVisualBlock(
        image_path=img_path,
        page_number=1,
        visual_index=1,
    )

    summary = summarizer.summarize_visual(visual, document_name="TestReport.pdf", use_cache=True)
    assert summary == "Pre-cached summary from disk"
    # Verify Groq client was never called because disk cache was hit!
    summarizer.client.chat.completions.create.assert_not_called()


def test_hierarchical_chunker_parent_child_linkage(mock_visual_summarizer, tmp_path):
    """Verify full hierarchical chunking creates valid parent-child relationships."""
    summarizer, asset_store = mock_visual_summarizer

    # Create valid dummy image on disk using PIL
    from PIL import Image
    img_path = tmp_path / "chart1.png"
    img = Image.new("RGB", (200, 200), color="blue")
    img.save(img_path, format="PNG")

    # Build mock parsed bundle
    bundle = ParsedDocumentBundle(
        source_name="NovaCore_Demo.pdf",
        document_hash="hash_demo_123",
        total_pages=1,
        pages=[
            ParsedPage(
                page_number=1,
                width=612.0,
                height=792.0,
                text_blocks=[
                    ExtractedTextBlock(text="NovaCore Systems provides enterprise AI infrastructure.", page_number=1),
                    ExtractedTextBlock(text="Founded in 2021, NovaCore operates globally.", page_number=1),
                ],
                tables=[
                    ExtractedTableBlock(
                        markdown="| Region | Revenue |\n| --- | --- |\n| North America | $55M |",
                        page_number=1,
                        table_number=1,
                        bbox=BoundingBox(x0=50, y0=200, x1=500, y1=300),
                    )
                ],
                visuals=[
                    ExtractedVisualBlock(
                        image_path=img_path,
                        page_number=1,
                        visual_index=1,
                        bbox=BoundingBox(x0=50, y0=350, x1=500, y1=600),
                    )
                ],
            )
        ],
    )

    chunker = HierarchicalChunker(visual_summarizer=summarizer)
    child_chunks, parent_docs = chunker.chunk_bundle(bundle)

    assert len(parent_docs) == 1
    parent = parent_docs[0]
    assert parent.source == "NovaCore_Demo.pdf"
    assert parent.page_number == 1
    assert "NovaCore Systems" in parent.full_content
    assert "| Region | Revenue |" in parent.full_content

    # We expect 3 child chunks: 1 text, 1 table, 1 visual
    assert len(child_chunks) == 3
    modalities = {c.metadata.modality for c in child_chunks}
    assert modalities == {ModalityType.TEXT, ModalityType.TABLE, ModalityType.VISUAL}

    # Verify all child chunks link to the parent document ID
    for child in child_chunks:
        assert child.metadata.parent_chunk_id == parent.parent_id
        assert child.chunk_id in parent.child_chunk_ids
        assert child.metadata.document_hash == "hash_demo_123"

    # Verify visual chunk holds image path
    visual_chunk = next(c for c in child_chunks if c.metadata.modality == ModalityType.VISUAL)
    assert visual_chunk.metadata.image_path == str(img_path)
    assert "Mocked visual summary" in visual_chunk.page_content
