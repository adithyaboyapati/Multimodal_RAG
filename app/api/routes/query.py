"""
Query and grounded multimodal synthesis endpoints.
"""
import time
from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_hybrid_retriever, get_multimodal_generator
from app.generation.multimodal_generator import MultimodalGenerator
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.query import QueryRequest
from app.schemas.response import LatencyBreakdown, RAGResponse

router = APIRouter(prefix="/api/v1", tags=["Query"])


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the multimodal RAG knowledge base",
)
async def query_multimodal_rag(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    generator: MultimodalGenerator = Depends(get_multimodal_generator),
):
    """
    Execute end-to-end multimodal RAG:
    1. Query understanding & decomposition
    2. Hybrid (Dense + BM25) vector retrieval in Pinecone
    3. Cross-Encoder reranking
    4. Token-budgeted context & visual evidence assembly
    5. Grounded synthesis with citations via Text LLM or Multimodal VLM
    """
    latency = LatencyBreakdown()
    start_total = time.perf_counter()

    # 1. Retrieval & Reranking Stage
    retrieved_chunks, analysis = retriever.retrieve(request, latency=latency)

    # 2. Multimodal Generation Stage
    response = generator.generate(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        intent=analysis.intent,
        max_visuals=request.max_visuals,
        latency=latency,
    )

    # Final latency harmonization
    response.latency.total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)

    # If user opted out of visual artifact payloads, strip data URIs to save network bandwidth
    if not request.return_visual_artifacts:
        for art in response.visual_artifacts:
            art.data_uri = None

    return response
