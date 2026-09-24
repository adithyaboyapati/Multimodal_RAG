"""
Unit tests for multimodal PDF parsing, table extraction, and visual asset extraction.
"""
from pathlib import Path
import pytest

from app.parsing.pdf_parser import PDFParser
from app.storage.asset_store import AssetStore

PDF_PATH = (
    Path("docs/NovaCore_Multimodal_Company_Report_2026.pdf")
    if Path("docs/NovaCore_Multimodal_Company_Report_2026.pdf").exists()
    else Path("NovaCore_Multimodal_Company_Report_2026.pdf")
)


@pytest.fixture
def asset_store(tmp_path):
    """Temporary asset store fixture for isolated tests."""
    from app.config.settings import Settings
    custom_settings = Settings(
        groq_api_key="mock_key",
        pinecone_api_key="mock_key",
        storage_dir=tmp_path / "test_assets",
    )
    return AssetStore(settings=custom_settings)


def test_asset_store_file_hash(asset_store):
    """Verify SHA-256 computation is deterministic."""
    assert PDF_PATH.exists()
    hash1 = asset_store.compute_file_hash(PDF_PATH)
    hash2 = asset_store.compute_file_hash(PDF_PATH)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 length


def test_pdf_parsing_pipeline(asset_store):
    """Verify full layout-aware extraction on NovaCore company report."""
    assert PDF_PATH.exists()
    parser = PDFParser(asset_store=asset_store)
    bundle = parser.parse(PDF_PATH)

    assert bundle.source_name == "NovaCore_Multimodal_Company_Report_2026.pdf"
    assert bundle.total_pages == 9
    assert len(bundle.pages) == 9

    # Aggregate statistics
    total_text_blocks = sum(len(p.text_blocks) for p in bundle.pages)
    total_tables = sum(len(p.tables) for p in bundle.pages)
    total_visuals = sum(len(p.visuals) for p in bundle.pages)

    assert total_text_blocks > 20
    assert total_tables >= 7
    assert total_visuals >= 6

    # Verify table formatting
    page_with_table = next(p for p in bundle.pages if len(p.tables) > 0)
    first_table = page_with_table.tables[0]
    assert "|" in first_table.markdown
    assert first_table.bbox.x1 > first_table.bbox.x0

    # Verify visual asset extraction and Data URI encoding
    page_with_visual = next(p for p in bundle.pages if len(p.visuals) > 0)
    first_visual = page_with_visual.visuals[0]
    assert first_visual.image_path.exists()

    data_uri = asset_store.image_to_data_uri(first_visual.image_path)
    assert data_uri.startswith("data:image/jpeg;base64,")
