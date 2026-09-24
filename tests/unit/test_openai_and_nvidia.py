"""
Unit tests for OpenAIDenseEmbedder (text-embedding-3-small) and NVIDIAReranker (NIM API).
"""
from unittest.mock import MagicMock, patch
import pytest

from app.config.constants import ModalityType
from app.config.settings import Settings
from app.embeddings.openai_embedder import OpenAIDenseEmbedder
from app.retrieval.nvidia_reranker import NVIDIAReranker
from app.schemas.document import DocumentMetadata
from app.schemas.query import RetrievedChunk


@pytest.fixture
def mock_settings():
    return Settings(
        groq_api_key="mock_groq",
        pinecone_api_key="mock_pinecone",
        openai_api_key="sk-mock-openai-key",
        nvidia_api_key="nvapi-mock-nvidia-key",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimension=384,
        reranker_provider="nvidia",
        nvidia_rerank_model="nvidia/reranking",
    )


def test_openai_dense_embedder_query_and_batch(mock_settings):
    """Verify OpenAIDenseEmbedder calls OpenAI embeddings API with dimensions parameter."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.05] * 384
    mock_response.data = [mock_item]
    mock_client.embeddings.create.return_value = mock_response

    embedder = OpenAIDenseEmbedder(
        settings=mock_settings,
        client=mock_client,
        dimension=384,
    )

    assert embedder.dimension == 384

    # 1. Embed Query
    query_vec = embedder.embed_query("What is revenue growth?")
    assert len(query_vec) == 384
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["What is revenue growth?"],
        dimensions=384,
    )

    # 2. Embed Documents
    mock_client.embeddings.create.reset_mock()
    docs = ["Doc 1 text", "Doc 2 text"]
    mock_item2 = MagicMock()
    mock_item2.embedding = [0.08] * 384
    mock_response.data = [mock_item, mock_item2]

    doc_vecs = embedder.embed_documents(docs)
    assert len(doc_vecs) == 2
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=docs,
        dimensions=384,
    )


@patch("requests.post")
def test_nvidia_reranker_reordering(mock_post, mock_settings):
    """Verify NVIDIAReranker calls NVIDIA NIM API and reorders candidate chunks."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "rankings": [
            {"index": 1, "logit": 3.45},
            {"index": 0, "logit": 1.12},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = NVIDIAReranker(
        settings=mock_settings,
        api_key="nvapi-mock-key",
    )

    chunk_a = RetrievedChunk(
        chunk_id="chunk-a",
        page_content="Overview of European operations",
        metadata=DocumentMetadata(source="doc.pdf", document_hash="h1", page_number=1, modality=ModalityType.TEXT),
        score=0.70,
        rank=1,
    )
    chunk_b = RetrievedChunk(
        chunk_id="chunk-b",
        page_content="Detailed revenue numbers by region",
        metadata=DocumentMetadata(source="doc.pdf", document_hash="h1", page_number=4, modality=ModalityType.TEXT),
        score=0.65,
        rank=2,
    )

    reranked = reranker.rerank(
        query="What are the regional revenue numbers?",
        chunks=[chunk_a, chunk_b],
        top_k=2,
    )

    assert len(reranked) == 2
    # chunk-b was index 1 with higher logit 3.45, so it should now be rank 1
    assert reranked[0].chunk_id == "chunk-b"
    assert reranked[0].rank == 1
    assert reranked[0].score == 3.45

    assert reranked[1].chunk_id == "chunk-a"
    assert reranked[1].rank == 2
    assert reranked[1].score == 1.12

    mock_post.assert_called_once()
    post_kwargs = mock_post.call_args[1]
    assert post_kwargs["json"]["model"] == "nvidia/reranking"
    assert len(post_kwargs["json"]["passages"]) == 2
