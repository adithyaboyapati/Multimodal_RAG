"""
Dependency injection container for FastAPI routes.
Provides cached singletons for pipeline components.
"""
from functools import lru_cache

from app.chunking.hierarchical import HierarchicalChunker
from app.config.settings import Settings, get_settings
from app.embeddings.base import BaseDenseEmbedder
from app.embeddings.dense import SentenceTransformerDenseEmbedder
from app.embeddings.openai_embedder import OpenAIDenseEmbedder
from app.generation.multimodal_generator import MultimodalGenerator
from app.parsing.pdf_parser import PDFParser
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.nvidia_reranker import NVIDIAReranker
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridVectorStore
from app.storage.asset_store import AssetStore, get_asset_store


@lru_cache()
def get_app_settings() -> Settings:
    """Provide application settings singleton."""
    return get_settings()


@lru_cache()
def get_dense_embedder() -> BaseDenseEmbedder:
    """
    Provide dense embedder singleton.
    Uses OpenAI text-embedding-3-small when configured with API key,
    otherwise falls back to local SentenceTransformers.
    """
    settings = get_app_settings()
    if (
        settings.embedding_provider == "openai"
        and settings.openai_api_key
        and not settings.openai_api_key.startswith("your_")
    ):
        return OpenAIDenseEmbedder(
            model_name=settings.embedding_model or "text-embedding-3-small",
            dimension=settings.embedding_dimension,
            settings=settings,
        )
    return SentenceTransformerDenseEmbedder(settings=settings)


@lru_cache()
def get_reranker():
    """
    Provide reranker singleton.
    Uses NVIDIA NeMo Retriever NIM API when configured with API key,
    otherwise falls back to local CrossEncoder.
    """
    settings = get_app_settings()
    if (
        settings.reranker_provider == "nvidia"
        and settings.nvidia_api_key
        and not settings.nvidia_api_key.startswith("your_")
    ):
        return NVIDIAReranker(settings=settings)
    return CrossEncoderReranker()


@lru_cache()
def get_pdf_parser() -> PDFParser:
    """Provide PDF parser singleton."""
    return PDFParser(asset_store=get_asset_store())


@lru_cache()
def get_hierarchical_chunker() -> HierarchicalChunker:
    """Provide multimodal hierarchical chunker singleton."""
    return HierarchicalChunker(settings=get_app_settings())


@lru_cache()
def get_vector_store() -> PineconeHybridVectorStore:
    """Provide Pinecone hybrid vector store singleton."""
    return PineconeHybridVectorStore(
        settings=get_app_settings(),
        dense_embedder=get_dense_embedder(),
    )


@lru_cache()
def get_hybrid_retriever() -> HybridRetriever:
    """Provide hybrid retriever singleton."""
    return HybridRetriever(
        settings=get_app_settings(),
        vector_store=get_vector_store(),
        reranker=get_reranker(),
    )


@lru_cache()
def get_multimodal_generator() -> MultimodalGenerator:
    """Provide multimodal generator singleton."""
    return MultimodalGenerator(settings=get_app_settings())
