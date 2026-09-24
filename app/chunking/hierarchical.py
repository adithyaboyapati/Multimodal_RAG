"""
Hierarchical Parent-Child chunking engine for multimodal enterprise documents.
Connects high-precision retrieval chunks (child) to contextual synthesis windows (parent).
"""
from typing import List, Optional, Tuple
from uuid import uuid4

from app.chunking.base import BaseChunker
from app.chunking.visual_summarizer import VisualSummarizer
from app.config.constants import ModalityType
from app.config.settings import Settings, get_settings
from app.parsing.base import ParsedDocumentBundle, ParsedPage
from app.schemas.document import DocumentChunk, DocumentMetadata, ParentDocument


class HierarchicalChunker(BaseChunker):
    """
    Creates fine-grained child chunks for dense/sparse vector indexing,
    while maintaining bidirectional links to parent page/section contexts.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        visual_summarizer: Optional[VisualSummarizer] = None,
    ):
        self.settings = settings or get_settings()
        self.visual_summarizer = visual_summarizer or VisualSummarizer(settings=self.settings)

    def _split_text(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Split a long text paragraph into overlapping character/token windows."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Try to break on newline or sentence boundary
            break_point = text.rfind("\n", start, end)
            if break_point == -1 or break_point <= start:
                break_point = text.rfind(". ", start, end)
            if break_point == -1 or break_point <= start:
                break_point = text.rfind(" ", start, end)
            if break_point == -1 or break_point <= start:
                break_point = end
            else:
                break_point += 1

            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, break_point - overlap_chars)

        return chunks

    def chunk_bundle(
        self, bundle: ParsedDocumentBundle
    ) -> Tuple[List[DocumentChunk], List[ParentDocument]]:
        """
        Process the entire parsed document into child chunks and parent contexts.
        """
        child_chunks: List[DocumentChunk] = []
        parent_documents: List[ParentDocument] = []
        global_chunk_index = 0

        for page in bundle.pages:
            parent_id = str(uuid4())
            child_ids_for_parent: List[str] = []

            # 1. Build Parent Context for this page
            parent_text_parts = [f"=== Page {page.page_number} ==="]
            for tb in page.text_blocks:
                parent_text_parts.append(tb.text)
            for tbl in page.tables:
                parent_text_parts.append(f"[Table {tbl.table_number}]\n{tbl.markdown}")

            full_parent_content = "\n\n".join(parent_text_parts)
            parent_doc = ParentDocument(
                parent_id=parent_id,
                source=bundle.source_name,
                page_number=page.page_number,
                full_content=full_parent_content,
                child_chunk_ids=[],
            )

            # 2. Child Chunks: Text Modality
            # Combine sequential text blocks into coherent paragraphs
            combined_page_text = "\n\n".join([tb.text for tb in page.text_blocks if tb.text.strip()])
            if combined_page_text:
                # Estimate chars: ~4 chars per token. chunk_size=400 tokens -> ~1600 chars
                max_chars = self.settings.chunk_size * 4
                overlap_chars = self.settings.chunk_overlap * 4
                text_splits = self._split_text(combined_page_text, max_chars, overlap_chars)

                for split_text in text_splits:
                    chunk_id = str(uuid4())
                    meta = DocumentMetadata(
                        source=bundle.source_name,
                        document_hash=bundle.document_hash,
                        page_number=page.page_number,
                        modality=ModalityType.TEXT,
                        parent_chunk_id=parent_id,
                        chunk_index=global_chunk_index,
                    )
                    child_chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        page_content=split_text,
                        metadata=meta,
                    )
                    child_chunks.append(child_chunk)
                    child_ids_for_parent.append(chunk_id)
                    global_chunk_index += 1

            # 3. Child Chunks: Table Modality
            for table_block in page.tables:
                chunk_id = str(uuid4())
                table_content = (
                    f"[Document: {bundle.source_name} | Page {page.page_number} | "
                    f"Table {table_block.table_number}]\n{table_block.markdown}"
                )
                meta = DocumentMetadata(
                    source=bundle.source_name,
                    document_hash=bundle.document_hash,
                    page_number=page.page_number,
                    modality=ModalityType.TABLE,
                    table_number=table_block.table_number,
                    parent_chunk_id=parent_id,
                    chunk_index=global_chunk_index,
                    bounding_box=table_block.bbox,
                )
                child_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    page_content=table_content,
                    metadata=meta,
                )
                child_chunks.append(child_chunk)
                child_ids_for_parent.append(chunk_id)
                global_chunk_index += 1

            # 4. Child Chunks: Visual Modality
            for visual_block in page.visuals:
                chunk_id = str(uuid4())
                summary = self.visual_summarizer.summarize_visual(
                    visual=visual_block,
                    document_name=bundle.source_name,
                )
                visual_content = (
                    f"[Document: {bundle.source_name} | Page {page.page_number} | "
                    f"Visual Asset {visual_block.visual_index}]\n{summary}"
                )
                meta = DocumentMetadata(
                    source=bundle.source_name,
                    document_hash=bundle.document_hash,
                    page_number=page.page_number,
                    modality=ModalityType.VISUAL,
                    image_path=str(visual_block.image_path),
                    parent_chunk_id=parent_id,
                    chunk_index=global_chunk_index,
                    bounding_box=visual_block.bbox,
                )
                child_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    page_content=visual_content,
                    metadata=meta,
                )
                child_chunks.append(child_chunk)
                child_ids_for_parent.append(chunk_id)
                global_chunk_index += 1

            parent_doc.child_chunk_ids = child_ids_for_parent
            parent_documents.append(parent_doc)

        return child_chunks, parent_documents
