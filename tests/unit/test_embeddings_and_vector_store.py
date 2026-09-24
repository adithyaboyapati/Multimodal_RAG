"""
Unit tests for dense embeddings, BM25 sparse embeddings, and Pinecone hybrid vector store.
"""
from unittest.mock import MagicMock
import pytest

from app.config.constants import ModalityType
from app.config.settings import Settings
from app.embeddings.dense import SentenceTransformerDenseEmbedder
from app.embeddings.sparse import BM25SparseEmbedder
from app.retrieval.vector_store import PineconeHybridVectorStore
from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.query import FilterCriteria


@pytest.fixture
def mock_settings():
    return Settings(
        groq_api_key="mock_groq",
        pinecone_api_key="mock_pinecone",
        pinecone_index_name="test-index",
        pinecone_namespace="test-namespace",
        hybrid_alpha=0.6,
    )


def test_dense_embedder():
    """Verify dense embedding generates 384-dim normalized vectors."""
    embedder = SentenceTransformerDenseEmbedder()
    assert embedder.dimension == 384

    query_vec = embedder.embed_query("NovaCore total revenue")
    assert len(query_vec) == 384
    assert isinstance(query_vec[0], float)

    docs = ["Page 1 text", "Page 2 text"]
    doc_vecs = embedder.embed_documents(docs)
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 384


def test_bm25_sparse_embedder():
    """Verify BM25 sparse encoding handles exact alphanumeric tokens and currencies."""
    sparse = BM25SparseEmbedder()
    text = "Product NC-942 achieved $132.0M in total FY2026 revenue."
    doc_sparse = sparse.encode_document(text)

    assert "indices" in doc_sparse
    assert "values" in doc_sparse
    assert len(doc_sparse["indices"]) == len(doc_sparse["values"])
    assert len(doc_sparse["indices"]) > 0

    # Verify indices are sorted and strictly non-negative 32-bit ints
    assert doc_sparse["indices"] == sorted(doc_sparse["indices"])
    assert all(0 <= idx < 2147483647 for idx in doc_sparse["indices"])

    # Verify query encoding
    query_sparse = sparse.encode_query("NC-942")
    assert len(query_sparse["indices"]) == 1
    assert query_sparse["values"][0] > 0.0


def test_pinecone_hybrid_vector_store_upsert_and_search(mock_settings):
    """Verify hybrid upsert formats dense + sparse vectors and search applies alpha weighting."""
    mock_client = MagicMock()
    mock_index = MagicMock()
    mock_client.has_index.return_value = True
    mock_client.Index.return_value = mock_index

    # Mock query search response
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "chunk-1",
                "score": 0.88,
                "metadata": {
                    "source": "NovaCore.pdf",
                    "document_hash": "hash_1",
                    "page": 3,
                    "modality": "table",
                    "text": "Revenue table content",
                    "table_number": 1,
                },
            }
        ]
    }

    mock_dense = MagicMock()
    mock_dense.dimension = 384
    mock_dense.embed_documents.return_value = [[0.1] * 384]
    mock_dense.embed_query.return_value = [0.2] * 384

    mock_sparse = MagicMock()
    mock_sparse.encode_documents.return_value = [{"indices": [100, 200], "values": [1.5, 2.0]}]
    mock_sparse.encode_query.return_value = {"indices": [100], "values": [1.5]}

    store = PineconeHybridVectorStore(
        settings=mock_settings,
        dense_embedder=mock_dense,
        sparse_embedder=mock_sparse,
        client=mock_client,
    )

    # 1. Test Upsert
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        page_content="Revenue table content",
        metadata=DocumentMetadata(
            source="NovaCore.pdf",
            document_hash="hash_1",
            page_number=3,
            modality=ModalityType.TABLE,
            table_number=1,
        ),
    )
    upserted_count = store.upsert_chunks([chunk])
    assert upserted_count == 1
    mock_index.upsert.assert_called_once()
    upsert_args = mock_index.upsert.call_args[1]
    assert upsert_args["namespace"] == "test-namespace"
    vector_record = upsert_args["vectors"][0]
    assert vector_record["id"] == "chunk-1"
    assert "sparse_values" in vector_record
    assert vector_record["sparse_values"]["indices"] == [100, 200]

    # 2. Test Hybrid Search with Alpha
    results = store.hybrid_search(
        query="What was revenue?",
        top_k=5,
        alpha=0.7,
        filters=FilterCriteria(pages=[3]),
    )
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.88
    assert results[0].metadata.modality == ModalityType.TABLE
    assert results[0].metadata.page_number == 3

    # Verify query arguments passed to Pinecone
    query_call_kwargs = mock_index.query.call_args[1]
    assert query_call_kwargs["top_k"] == 5
    # Scaled dense: 0.2 * 0.7 = 0.14
    assert pytest.approx(query_call_kwargs["vector"][0], 0.001) == 0.14
    # Scaled sparse: 1.5 * (1 - 0.7) = 0.45
    assert pytest.approx(query_call_kwargs["sparse_vector"]["values"][0], 0.001) == 0.45
    assert query_call_kwargs["filter"] == {"page": {"$in": [3]}}

    # 3. Test Non-Destructive Delete Document
    store.delete_document("hash_1")
    mock_index.delete.assert_called_once_with(
        filter={"document_hash": {"$eq": "hash_1"}},
        namespace="test-namespace",
    )
