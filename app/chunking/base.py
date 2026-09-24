"""
Abstract interface for multimodal chunking and representation strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from app.parsing.base import ParsedDocumentBundle
from app.schemas.document import DocumentChunk, ParentDocument


class BaseChunker(ABC):
    """Abstract interface defining the contract for multimodal document chunkers."""

    @abstractmethod
    def chunk_bundle(
        self, bundle: ParsedDocumentBundle
    ) -> Tuple[List[DocumentChunk], List[ParentDocument]]:
        """
        Process a parsed multimodal document bundle into retrievable child chunks
        and contextual parent documents.

        Returns:
            Tuple of (child_chunks, parent_documents)
        """
        pass
