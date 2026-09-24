"""
High-level hybrid retriever orchestrating query analysis, parallel sub-query retrieval,
Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
"""
from typing import Dict, List, Optional, Tuple

from app.config.settings import Settings, get_settings
from app.config.tracing import traceable
from app.retrieval.query_analyzer import QueryAnalysisResult, QueryAnalyzer
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridVectorStore
from app.schemas.query import FilterCriteria, QueryRequest, RetrievedChunk
from app.schemas.response import LatencyBreakdown


class HybridRetriever:
    """
    Production retriever coordinating:
    - Query understanding & intent classification
    - Sub-query decomposition for multi-part questions
    - Pinecone hybrid (Dense + BM25) search
    - Reciprocal Rank Fusion (RRF) deduplication
    - Cross-modal cross-encoder reranking
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[PineconeHybridVectorStore] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store or PineconeHybridVectorStore(settings=self.settings)
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        if reranker is not None:
            self.reranker = reranker
        elif (
            self.settings.reranker_provider == "nvidia"
            and self.settings.nvidia_api_key
            and not self.settings.nvidia_api_key.startswith("your_")
        ):
            from app.retrieval.nvidia_reranker import NVIDIAReranker
            self.reranker = NVIDIAReranker(settings=self.settings)
        else:
            self.reranker = CrossEncoderReranker()

    def _reciprocal_rank_fusion(
        self,
        query_results: List[List[RetrievedChunk]],
        k_constant: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).
        Score(d) = Sum( 1 / (k + rank(d)) )
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        for result_list in query_results:
            for rank, chunk in enumerate(result_list, start=1):
                chunk_id = chunk.chunk_id
                chunk_map[chunk_id] = chunk
                score = 1.0 / (k_constant + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        # Sort chunks by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_chunks = []
        for rank, cid in enumerate(sorted_ids, start=1):
            chunk = chunk_map[cid]
            chunk.score = rrf_scores[cid]
            chunk.rank = rank
            fused_chunks.append(chunk)

        return fused_chunks

    @traceable(name="hybrid_retrieval", run_type="retriever")
    def retrieve(
        self,
        request: QueryRequest,
        latency: Optional[LatencyBreakdown] = None,
    ) -> Tuple[List[RetrievedChunk], QueryAnalysisResult]:
        """
        Execute full retrieval pipeline for an incoming user query request.
        """
        import time

        # 1. Analyze query intent and extract sub-queries/filters
        t_intent_start = time.perf_counter()
        analysis = self.query_analyzer.analyze_query(request.question)
        if latency:
            latency.intent_ms = round((time.perf_counter() - t_intent_start) * 1000.0, 2)

        # Merge explicit request filters with analyzer-extracted filters
        effective_filters = request.filters
        if analysis.extracted_filters and not effective_filters:
            effective_filters = analysis.extracted_filters

        # 2. Retrieve candidates from Pinecone
        # Fetch an optimal candidate pool (6 to 8 chunks) for reranking to minimize payload & latency
        fetch_k = min(max(request.top_k + 2, 6), 8)

        t_retrieval_start = time.perf_counter()
        if len(analysis.sub_queries) > 1:
            # Parallel multi-query execution: retrieve per sub-query concurrently and fuse via RRF
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(analysis.sub_queries), 4)) as executor:
                futures = [
                    executor.submit(
                        self.vector_store.hybrid_search,
                        query=sub_q,
                        top_k=fetch_k,
                        alpha=request.hybrid_alpha,
                        filters=effective_filters,
                    )
                    for sub_q in analysis.sub_queries
                ]
                multi_results = [f.result() for f in futures]

            candidate_chunks = self._reciprocal_rank_fusion(multi_results)
        else:
            # Single-query execution
            candidate_chunks = self.vector_store.hybrid_search(
                query=request.question,
                top_k=fetch_k,
                alpha=request.hybrid_alpha,
                filters=effective_filters,
            )
        if latency:
            latency.retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000.0, 2)

        # 3. Cross-Encoder Reranking
        t_rerank_start = time.perf_counter()
        if request.rerank and candidate_chunks:
            final_chunks = self.reranker.rerank(
                query=request.question,
                chunks=candidate_chunks,
                top_k=request.top_k,
            )
        else:
            final_chunks = candidate_chunks[: request.top_k]
        if latency:
            latency.rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)

        return final_chunks, analysis
