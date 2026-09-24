"""
Token-budgeted context assembler and visual artifact serializer.
Formats retrieved evidence, serializes image assets, and builds candidate citations.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.config.constants import ModalityType
from app.schemas.query import RetrievedChunk
from app.schemas.response import Citation, VisualArtifact
from app.storage.asset_store import AssetStore, get_asset_store

logger = logging.getLogger("novacore.generation.context_builder")

DEFAULT_MAX_CONTEXT_CHARS = 16000  # ~4000 tokens for context text


@dataclass
class AssembledContext:
    """The structured result of context assembly."""
    context_text: str
    visual_artifacts: List[VisualArtifact] = field(default_factory=list)
    candidate_citations: List[Citation] = field(default_factory=list)


class ContextBuilder:
    """
    Assembles retrieved text, table, and visual evidence into token-budgeted prompt context.
    Serializes top visual assets into Base64 Data URIs for VLM consumption.
    """

    def __init__(self, asset_store: Optional[AssetStore] = None):
        self.asset_store = asset_store or get_asset_store()

    def build_context(
        self,
        chunks: List[RetrievedChunk],
        max_visuals: int = 3,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> AssembledContext:
        """
        Format retrieved chunks into a unified context string and prepare attached visuals.
        """
        text_parts: List[str] = []
        visual_artifacts: List[VisualArtifact] = []
        candidate_citations: List[Citation] = []
        current_chars = 0

        # Sort chunks by rank to prioritize highest-scoring evidence
        sorted_chunks = sorted(chunks, key=lambda c: c.rank)

        for chunk in sorted_chunks:
            meta = chunk.metadata
            page_num = meta.page_number
            modality = meta.modality

            # 1. Format context header with unambiguous citation provenance
            header = f"[Page {page_num} | {modality.value.upper()} | Source: {meta.source}]"
            entry = f"{header}\n{chunk.page_content.strip()}"

            # 2. Check token/character budget
            if current_chars + len(entry) <= max_context_chars:
                text_parts.append(entry)
                current_chars += len(entry) + 2

            # 3. Create Candidate Citation
            excerpt = chunk.page_content[:200].replace("\n", " ").strip()
            citation = Citation(
                source=meta.source,
                page_number=page_num,
                modality=modality,
                excerpt=excerpt,
                bounding_box=meta.bounding_box,
                confidence_score=chunk.score,
            )
            candidate_citations.append(citation)

            # 4. Process Visual Assets for Multimodal Generation
            if modality == ModalityType.VISUAL and len(visual_artifacts) < max_visuals:
                if meta.image_path:
                    img_path = Path(meta.image_path)
                    if img_path.exists():
                        try:
                            data_uri = self.asset_store.image_to_data_uri(img_path)
                            artifact = VisualArtifact(
                                asset_id=chunk.chunk_id,
                                page_number=page_num,
                                image_path=str(img_path),
                                data_uri=data_uri,
                                summary=chunk.page_content,
                            )
                            visual_artifacts.append(artifact)
                        except Exception as exc:
                            logger.debug("Failed serializing visual asset %s: %s", img_path, exc)

        formatted_context_text = "\n\n".join(text_parts)

        return AssembledContext(
            context_text=formatted_context_text,
            visual_artifacts=visual_artifacts,
            candidate_citations=candidate_citations,
        )
