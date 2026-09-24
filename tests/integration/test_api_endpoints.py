"""
Integration tests for FastAPI endpoints (/health, /api/v1/ingest, /api/v1/query).
"""
import io
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_app_settings,
    get_hierarchical_chunker,
    get_hybrid_retriever,
    get_multimodal_generator,
    get_pdf_parser,
    get_vector_store,
)
from app.api.server import create_app
from app.config.constants import ModalityType
from app.config.settings import Settings
from app.parsing.base import ParsedDocumentBundle, ParsedPage
from app.retrieval.query_analyzer import QueryAnalysisResult
from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.query import ModalityIntent, RetrievedChunk
from app.schemas.response import Citation, LatencyBreakdown, RAGResponse


@pytest.fixture
def client_and_mocks():
    """Create FastAPI TestClient with mocked service dependencies."""
    settings = Settings(
        groq_api_key="mock_groq_key",
        pinecone_api_key="mock_pinecone_key",
    )

    mock_parser = MagicMock()
    mock_parser.parse.return_value = ParsedDocumentBundle(
        source_name="test.pdf",
        document_hash="test_hash_123",
        total_pages=2,
        pages=[ParsedPage(page_number=1, width=612, height=792), ParsedPage(page_number=2, width=612, height=792)],
    )

    mock_chunker = MagicMock()
    chunk1 = DocumentChunk(
        chunk_id="c1",
        page_content="Text content",
        metadata=DocumentMetadata(source="test.pdf", page_number=1, modality=ModalityType.TEXT),
    )
    mock_chunker.chunk_bundle.return_value = ([chunk1], [])

    mock_store = MagicMock()
    mock_store.upsert_chunks.return_value = 1

    mock_retriever = MagicMock()
    retrieved_chunk = RetrievedChunk(
        chunk_id="c1",
        page_content="NovaCore revenue reached $132M.",
        metadata=DocumentMetadata(source="test.pdf", page_number=2, modality=ModalityType.TEXT),
        score=0.91,
        rank=1,
    )
    analysis = QueryAnalysisResult(
        original_query="What was revenue?",
        intent=ModalityIntent.TEXT_ONLY,
        sub_queries=["What was revenue?"],
    )
    mock_retriever.retrieve.return_value = ([retrieved_chunk], analysis)

    mock_generator = MagicMock()
    mock_generator.generate.return_value = RAGResponse(
        question="What was revenue?",
        answer="Revenue was $132M [Page 2 | TEXT].",
        citations=[Citation(source="test.pdf", page_number=2, modality=ModalityType.TEXT, excerpt="revenue reached $132M")],
        visual_artifacts=[],
        retrieved_chunks_count=1,
        modality_used="text_llm",
        model="openai/gpt-oss-20b",
        latency=LatencyBreakdown(retrieval_ms=15.0, generation_ms=120.0, total_ms=135.0),
    )

    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_pdf_parser] = lambda: mock_parser
    app.dependency_overrides[get_hierarchical_chunker] = lambda: mock_chunker
    app.dependency_overrides[get_vector_store] = lambda: mock_store
    app.dependency_overrides[get_hybrid_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_multimodal_generator] = lambda: mock_generator

    return TestClient(app)


def test_health_live_endpoint(client_and_mocks):
    """Verify /health/live returns 200 OK."""
    client = client_and_mocks
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "live"


def test_health_ready_endpoint(client_and_mocks):
    """Verify /health/ready returns 200 OK when keys configured."""
    client = client_and_mocks
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_query_endpoint(client_and_mocks):
    """Verify /api/v1/query returns valid RAGResponse payload."""
    client = client_and_mocks
    payload = {
        "question": "What was total revenue in FY2026?",
        "top_k": 3,
        "hybrid_alpha": 0.7,
    }
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "What was revenue?"
    assert "Revenue was $132M" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["page_number"] == 2
    assert data["modality_used"] == "text_llm"


def test_ingest_endpoint_valid_and_invalid(client_and_mocks):
    """Verify /api/v1/ingest handles file upload and rejects invalid file types."""
    client = client_and_mocks

    # 1. Invalid file type (.txt)
    fake_txt = io.BytesIO(b"Not a PDF")
    resp_invalid = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", fake_txt, "text/plain")},
    )
    assert resp_invalid.status_code == 400

    # 2. Valid PDF file
    fake_pdf = io.BytesIO(b"%PDF-1.4 mock pdf content")
    resp_valid = client.post(
        "/api/v1/ingest",
        files={"file": ("report.pdf", fake_pdf, "application/pdf")},
    )
    assert resp_valid.status_code == 201
    summary = resp_valid.json()
    assert summary["document_name"] == "report.pdf"
    assert summary["total_pages"] == 2
    assert summary["total_vectors_indexed"] == 1
