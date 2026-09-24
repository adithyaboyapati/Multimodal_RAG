"""
Unified multimodal generator orchestrating grounded text and vision inference with Groq.
Measures latency breakdown and resolves verified citations.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from groq import Groq

logger = logging.getLogger("novacore.generation.multimodal")

from app.config.constants import PromptTemplates
from app.config.settings import Settings, get_settings
from app.generation.context_builder import ContextBuilder
from app.schemas.query import ModalityIntent, RetrievedChunk
from app.schemas.response import Citation, LatencyBreakdown, RAGResponse, VisualArtifact


class MultimodalGenerator:
    """
    Production generator coordinating:
    - Token-budgeted context formatting
    - Dynamic routing (Text LLM vs Multimodal VLM)
    - Structured inline citations
    - Millisecond latency telemetry
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        context_builder: Optional[ContextBuilder] = None,
        client: Optional[Groq] = None,
    ):
        self.settings = settings or get_settings()
        self.context_builder = context_builder or ContextBuilder()
        self.client = client or Groq(api_key=self.settings.groq_api_key)

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        intent: Optional[ModalityIntent] = None,
        max_visuals: int = 2,
        latency: Optional[LatencyBreakdown] = None,
    ) -> RAGResponse:
        """
        Synthesize a factually grounded answer using retrieved chunks and visual evidence.
        """
        timing = latency or LatencyBreakdown()
        gen_start = time.perf_counter()

        # 1. Assemble context and serialize visual artifacts
        assembled = self.context_builder.build_context(
            chunks=retrieved_chunks,
            max_visuals=max_visuals,
        )

        visual_artifacts = assembled.visual_artifacts
        has_visuals = len(visual_artifacts) > 0

        # Smart Dynamic Routing:
        # Only invoke heavy multimodal vision model if the query specifically involves visual reasoning
        # (e.g. VISUAL or CROSS_MODAL intent) AND visual artifacts are present.
        # For TEXT_ONLY and TABULAR questions, use the high-speed Text LLM with retrieved text context
        # (which already includes the pre-computed visual summary of any charts).
        use_vision = has_visuals and (
            intent in (ModalityIntent.VISUAL, ModalityIntent.CROSS_MODAL)
            or intent is None
        )

        answer = ""
        modality_used = "text_llm"
        model_used = self.settings.text_model

        if use_vision:
            try:
                # Multimodal Vision Pathway (Qwen 3.8 27B Vision)
                # Keep prompt context concise (~2,000 chars) to minimize pre-fill latency for vision models
                vlm_context = assembled.context_text[:2500]
                system_prompt = PromptTemplates.MULTIMODAL_RAG_SYSTEM.format(
                    context=vlm_context
                ).strip()

                user_content: List[Dict[str, Any]] = [
                    {"type": "text", "text": f"QUESTION:\n{question}\n\nANSWER:"}
                ]

                # Attach base64 data URIs for top reranked images (capped to max 2)
                for artifact in visual_artifacts[:2]:
                    if artifact.data_uri:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": artifact.data_uri},
                            }
                        )

                response = self.client.chat.completions.create(
                    model=self.settings.vision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_completion_tokens=350,
                )
                answer = response.choices[0].message.content.strip()
                modality_used = "multimodal_vlm"
                model_used = self.settings.vision_model
            except Exception as exc:
                logger.warning("VLM generation failed or timed out: %s. Falling back to Text LLM.", exc)
                # Fallback to Text LLM if Vision Model encounters rate limit or network issue
                use_vision = False

        if not use_vision:
            # Fast Text LLM Pathway (GPT-OSS 20B)
            system_prompt = PromptTemplates.TEXT_RAG_SYSTEM.format(
                context=assembled.context_text
            ).strip()

            response = self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"QUESTION:\n{question}\n\nANSWER:"},
                ],
                temperature=0.0,
                max_completion_tokens=400,
            )
            raw_content = response.choices[0].message.content or ""
            if not raw_content and getattr(response.choices[0].message, "reasoning", None):
                raw_content = response.choices[0].message.reasoning
            answer = raw_content.strip()
            modality_used = "text_llm"
            model_used = self.settings.text_model

        # 3. Telemetry Computation
        timing.generation_ms = round((time.perf_counter() - gen_start) * 1000.0, 2)
        timing.total_ms = round(
            timing.intent_ms + timing.retrieval_ms + timing.rerank_ms + timing.generation_ms,
            2,
        )

        # 4. Citation Deduplication & Verification
        # Retain candidate citations that correspond to retrieved evidence
        seen_citations = set()
        deduped_citations: List[Citation] = []
        for cite in assembled.candidate_citations:
            key = (cite.source, cite.page_number, cite.modality)
            if key not in seen_citations:
                seen_citations.add(key)
                deduped_citations.append(cite)

        return RAGResponse(
            question=question,
            answer=answer,
            citations=deduped_citations,
            visual_artifacts=visual_artifacts,
            retrieved_chunks_count=len(retrieved_chunks),
            modality_used=modality_used,
            model=model_used,
            latency=timing,
        )
