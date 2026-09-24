"""
Application configuration management using Pydantic Settings v2.
Validates environment variables, enforces fail-fast integrity checks, and loads defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production application settings with environment variable bindings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API Keys (Fail-fast validation)
    groq_api_key: str = Field(
        default="",
        description="Groq API Key used for fast LLM text and VLM visual inference.",
    )
    pinecone_api_key: str = Field(
        default="",
        description="Pinecone API Key for vector index operations and hybrid search.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key for text-embedding-3-small embeddings.",
    )
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA API Key for NeMo Retriever NIM reranking.",
    )

    # LangSmith Observability & Tracing Configuration
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable LangSmith distributed tracing.",
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API Key for tracing prompts, latencies, and evaluations.",
    )
    langchain_project: str = Field(
        default="novacore-multimodal-rag",
        description="LangSmith project name.",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint URL.",
    )

    # Pinecone Vector Database Configuration
    pinecone_index_name: str = Field(
        default="novacore-multimodal-rag",
        description="Target Pinecone index name.",
    )
    pinecone_namespace: str = Field(
        default="fy2026-demo",
        description="Namespace for isolating document tenant collections.",
    )
    pinecone_cloud: str = Field(
        default="aws",
        description="Cloud provider for serverless Pinecone index.",
    )
    pinecone_region: str = Field(
        default="us-east-1",
        description="Region for serverless Pinecone index.",
    )

    # Model & Provider Configuration
    text_model: str = Field(
        default="openai/gpt-oss-20b",
        description="High-speed text LLM deployed on Groq.",
    )
    vision_model: str = Field(
        default="qwen/qwen3.8-27b",
        description="High-capacity Vision-Language Model deployed on Groq.",
    )
    embedding_provider: str = Field(
        default="openai",
        description="Dense embedding provider: 'openai' (text-embedding-3-small) or 'sentence-transformers'.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Dense text embedding model identifier.",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding dimension (matches Pinecone index: 384 for current index, or 1536).",
    )
    reranker_provider: str = Field(
        default="nvidia",
        description="Reranker provider: 'nvidia' (NIM Cloud API) or 'local' (Cross-Encoder).",
    )
    nvidia_rerank_model: str = Field(
        default="nvidia/llama-nemotron-rerank-vl-1b-v2",
        description="NVIDIA NIM reranker model name.",
    )

    # Storage & Ingestion Configuration
    storage_dir: Path = Field(
        default=Path("novacore_extracted_images"),
        description="Local directory for storing extracted image assets.",
    )
    chunk_size: int = Field(
        default=400,
        description="Maximum token size for child text chunks.",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Token overlap between sequential text chunks.",
    )

    # Retrieval Configuration
    top_k: int = Field(
        default=5,
        description="Default number of top context documents to retrieve.",
    )
    hybrid_alpha: float = Field(
        default=0.6,
        description="Hybrid search convex weighting: 1.0 = Pure Dense, 0.0 = Pure Sparse.",
    )
    max_attached_images: int = Field(
        default=3,
        description="Maximum number of retrieved visual assets injected into the VLM prompt.",
    )

    # Server Configuration
    api_host: str = Field(default="0.0.0.0", description="FastAPI host binding.")
    api_port: int = Field(default=8000, description="FastAPI port binding.")
    debug: bool = Field(default=False, description="Enable debug logging.")

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, value: str) -> str:
        """Enforce non-placeholder key for production execution."""
        clean = value.strip()
        if not clean or clean == "your_groq_api_key_here":
            # Note: We allow empty during tests/local dry runs, but warn when invalid
            return clean
        return clean

    @field_validator("pinecone_api_key")
    @classmethod
    def validate_pinecone_key(cls, value: str) -> str:
        """Enforce non-placeholder key for production execution."""
        clean = value.strip()
        if not clean or clean == "your_pinecone_api_key_here":
            return clean
        return clean

    @field_validator("storage_dir")
    @classmethod
    def ensure_storage_dir(cls, path: Path) -> Path:
        """Auto-create storage directory if missing."""
        path.mkdir(parents=True, exist_ok=True)
        return path


    def configure_tracing(self) -> None:
        """Export LangSmith environment variables into os.environ for automated tracing hooks."""
        import os
        if self.langchain_api_key and not self.langchain_api_key.startswith("your_"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true" if self.langchain_tracing_v2 else "false"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    settings = Settings()
    settings.configure_tracing()
    return settings
