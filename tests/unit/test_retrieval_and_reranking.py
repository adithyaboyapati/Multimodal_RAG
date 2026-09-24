"""
Unit tests for query analysis, sub-query decomposition, RRF fusion, and cross-encoder reranking.
"""
from unittest.mock import MagicMock
import pytest

from app.config.constants import ModalityType
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.document import DocumentMetadata
from app.schemas.query import ModalityIntent, QueryRequest, RetrievedChunk


def test_query_analyzer_intent_and_sub_queries():
    """Verify query analyzer accurately detects visual, tabular, and cross-modal intents."""
    analyzer = QueryAnalyzer()

    # Visual intent
    res1 = analyzer.analyze_query("According to the revenue graph, what was peak revenue?")
    assert res1.intent == ModalityIntent.VISUAL
    assert len(res1.sub_queries) == 1

    # Tabular intent
    res2 = analyzer.analyze_query("Show me the financial breakdown table for 2026.")
    assert res2.intent == ModalityIntent.TABULAR

    # Cross-modal multi-part query decomposition
    multi_part_query = """
Give me a short FY2026 performance summary.
Include:
1. total revenue
2. fastest-growing region
3. support-resolution improvement
"""
    res3 = analyzer.analyze_query(multi_part_query)
    assert res3.intent == ModalityIntent.CROSS_MODAL
    assert len(res3.sub_queries) == 3
    assert "total revenue" in res3.sub_queries[0]
    assert "fastest-growing region" in res3.sub_queries[1]

    # Explicit page filter extraction
    res4 = analyzer.analyze_query("What happened on page 7?")
    assert res4.extracted_filters is not None
    assert res4.extracted_filters.pages == [7]


def test_cross_encoder_reranker_reordering():
    """Verify reranker updates scores and properly sorts candidate chunks."""
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    # Mock scores: chunk B gets higher score (0.95) than chunk A (0.30)
    mock_model.predict.return_value = [0.30, 0.95]
    reranker._model = mock_model

    chunk_a = RetrievedChunk(
        chunk_id="chunk-a",
        page_content="Text about general operations",
        metadata=DocumentMetadata(source="doc.pdf", page_number=1, modality=ModalityType.TEXT),
        score=0.85,
        rank=1,
    )
    chunk_b = RetrievedChunk(
        chunk_id="chunk-b",
        page_content="Graph showing exact revenue peak at $42.5M",
        metadata=DocumentMetadata(source="doc.pdf", page_number=3, modality=ModalityType.VISUAL),
        score=0.70,
        rank=2,
    )

    reranked = reranker.rerank(
        query="What was peak revenue?",
        chunks=[chunk_a, chunk_b],
        top_k=2,
    )

    # Chunk B should now be #1 because of cross-encoder score
    assert reranked[0].chunk_id == "chunk-b"
    assert reranked[0].rank == 1
    assert reranked[0].score == 0.95
    assert reranked[1].chunk_id == "chunk-a"
    assert reranked[1].rank == 2


def test_hybrid_retriever_multi_query_and_rrf():
    """Verify retriever orchestrates sub-query retrieval and Reciprocal Rank Fusion."""
    mock_store = MagicMock()
    mock_analyzer = MagicMock()
    mock_reranker = MagicMock()

    # Create dummy retrieved chunks
    c1 = RetrievedChunk(
        chunk_id="c1",
        page_content="Doc 1",
        metadata=DocumentMetadata(source="d.pdf", page_number=1, modality=ModalityType.TEXT),
        score=0.9,
        rank=1,
    )
    c2 = RetrievedChunk(
        chunk_id="c2",
        page_content="Doc 2",
        metadata=DocumentMetadata(source="d.pdf", page_number=2, modality=ModalityType.TABLE),
        score=0.8,
        rank=2,
    )

    # Mock multi-query decomposition: 2 sub-queries
    from app.retrieval.query_analyzer import QueryAnalysisResult
    mock_analyzer.analyze_query.return_value = QueryAnalysisResult(
        original_query="Complex question",
        intent=ModalityIntent.CROSS_MODAL,
        sub_queries=["Sub query 1", "Sub query 2"],
        extracted_filters=None,
    )

    # Sub-query 1 returns [c1, c2], sub-query 2 returns [c2]
    mock_store.hybrid_search.side_effect = [[c1, c2], [c2]]
    # Reranker returns whatever is passed sliced to top_k
    mock_reranker.rerank.side_effect = lambda query, chunks, top_k: chunks[:top_k]

    retriever = HybridRetriever(
        vector_store=mock_store,
        query_analyzer=mock_analyzer,
        reranker=mock_reranker,
    )

    request = QueryRequest(question="Complex question", top_k=2)
    final_chunks, analysis = retriever.retrieve(request)

    # Both sub-queries executed
    assert mock_store.hybrid_search.call_count == 2
    # Reranker called with fused RRF candidate list
    mock_reranker.rerank.assert_called_once()
    assert len(final_chunks) <= 2
