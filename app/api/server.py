"""
FastAPI application factory, middleware, and lifecycle configuration.
"""
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, ingest, query
from app.config.settings import get_settings

logger = logging.getLogger("novacore.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm ML models, vector database, reranker, and LLM to eliminate first-query cold start."""
    try:
        from app.api.dependencies import get_hybrid_retriever, get_multimodal_generator
        from app.schemas.document import DocumentMetadata
        from app.schemas.query import RetrievedChunk

        logger.info("Pre-warming Multimodal RAG pipeline components...")
        retriever = get_hybrid_retriever()
        generator = get_multimodal_generator()

        # 1. Warm dense embedder
        try:
            _ = retriever.vector_store.dense_embedder.embed_query("warmup query")
            logger.info("Dense embedder pre-warmed.")
        except Exception as e:
            logger.warning(f"Embedder warmup warning: {e}")

        # 2. Warm Pinecone index connection and data plane session
        try:
            _ = retriever.vector_store.metric
            dim = retriever.vector_store.dense_embedder.dimension
            _ = retriever.vector_store.index.query(
                vector=[0.0] * dim,
                top_k=1,
                namespace=retriever.settings.pinecone_namespace,
            )
            logger.info("Pinecone vector store data plane connection pre-warmed.")
        except Exception as e:
            logger.warning(f"Pinecone warmup warning: {e}")

        # 3. Warm reranker (local or NVIDIA NIM cloud API)
        try:
            from app.config.constants import ModalityType
            sample_chunk = RetrievedChunk(
                chunk_id="warmup_1",
                page_content="warmup passage for pipeline preheating",
                metadata=DocumentMetadata(
                    source="warmup",
                    document_hash="",
                    page_number=1,
                    modality=ModalityType.TEXT,
                ),
                score=0.5,
                rank=1,
            )
            _ = retriever.reranker.rerank("warmup query", [sample_chunk], top_k=1)
            logger.info("Reranker pre-warmed.")
        except Exception as e:
            logger.warning(f"Reranker warmup warning: {e}")

        # 4. Warm Groq LLM connection
        try:
            _ = generator.client.chat.completions.create(
                model=generator.settings.text_model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=5,
            )
            logger.info("Text LLM client connection pre-warmed.")
        except Exception as e:
            logger.warning(f"Generator warmup warning: {e}")

        logger.info("Multimodal RAG pipeline is fully warm and ready for sub-second queries.")
    except Exception as exc:
        logger.warning(f"Startup pre-warm skipped or failed: {exc}")
    yield


def create_app() -> FastAPI:
    """Create and configure the production FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="NovaCore Multimodal Enterprise RAG",
        description=(
            "Production-grade Multimodal Retrieval-Augmented Generation (RAG) platform "
            "for enterprise reports containing text, markdown tables, and visual charts."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS) Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Global Exception Handlers
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Resource Not Found", "detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation Error", "detail": str(exc)},
        )

    # Register Route Handlers
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    return app


# Application entry point for ASGI servers (Uvicorn / Gunicorn)
app = create_app()
