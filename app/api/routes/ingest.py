"""
Document ingestion endpoints for parsing, chunking, summarizing, and indexing documents.
"""
import shutil
import tempfile
import time
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_app_settings,
    get_hierarchical_chunker,
    get_pdf_parser,
    get_vector_store,
)
from app.chunking.hierarchical import HierarchicalChunker
from app.config.constants import ModalityType
from app.config.settings import Settings
from app.parsing.pdf_parser import PDFParser
from app.retrieval.vector_store import PineconeHybridVectorStore
from app.schemas.document import IngestionSummary

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and index a multimodal document",
)
async def ingest_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_app_settings),
    parser: PDFParser = Depends(get_pdf_parser),
    chunker: HierarchicalChunker = Depends(get_hierarchical_chunker),
    vector_store: PineconeHybridVectorStore = Depends(get_vector_store),
):
    """
    Upload a PDF document, extract text/tables/visuals, generate VLM summaries,
    construct parent-child chunks, and index into Pinecone hybrid vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Currently, only PDF documents (.pdf) are supported for ingestion.",
        )

    start_time = time.perf_counter()

    # Save uploaded file to secure temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # 1. Parse document into multimodal bundle
        bundle = parser.parse(tmp_path)
        # Preserve original filename
        bundle.source_name = file.filename

        # 2. Hierarchical Chunking & Visual Summarization
        child_chunks, parent_docs = chunker.chunk_bundle(bundle)

        # 3. Upsert into Pinecone hybrid store
        upserted_count = vector_store.upsert_chunks(child_chunks)

        duration = round(time.perf_counter() - start_time, 2)

        # Compute modality counts
        text_count = sum(1 for c in child_chunks if c.metadata.modality == ModalityType.TEXT)
        table_count = sum(1 for c in child_chunks if c.metadata.modality == ModalityType.TABLE)
        image_count = sum(1 for c in child_chunks if c.metadata.modality == ModalityType.VISUAL)

        return IngestionSummary(
            document_name=file.filename,
            document_hash=bundle.document_hash,
            total_pages=bundle.total_pages,
            text_chunks_count=text_count,
            tables_count=table_count,
            images_count=image_count,
            total_vectors_indexed=upserted_count,
            duration_seconds=duration,
        )

    finally:
        # Cleanup temporary uploaded file
        if tmp_path.exists():
            tmp_path.unlink()
