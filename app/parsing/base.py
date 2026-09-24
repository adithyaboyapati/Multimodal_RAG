"""
Abstract base classes and intermediate representations for multimodal document parsing.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.schemas.document import BoundingBox


@dataclass
class ExtractedTextBlock:
    """A clean semantic text block extracted from a document page."""
    text: str
    page_number: int
    bbox: Optional[BoundingBox] = None


@dataclass
class ExtractedTableBlock:
    """A structured table extracted and converted to tabular representations."""
    markdown: str
    page_number: int
    table_number: int
    bbox: BoundingBox


@dataclass
class ExtractedVisualBlock:
    """A visual asset (chart, graph, diagram, or photo) extracted from a page."""
    image_path: Path
    page_number: int
    visual_index: int
    bbox: Optional[BoundingBox] = None
    is_vector_graphic: bool = False


@dataclass
class ParsedPage:
    """Aggregated multimodal extractions for an individual page."""
    page_number: int
    width: float
    height: float
    text_blocks: List[ExtractedTextBlock] = field(default_factory=list)
    tables: List[ExtractedTableBlock] = field(default_factory=list)
    visuals: List[ExtractedVisualBlock] = field(default_factory=list)
    is_scanned: bool = False
    raw_character_count: int = 0


@dataclass
class ParsedDocumentBundle:
    """The complete extraction output for an ingested document."""
    source_name: str
    document_hash: str
    total_pages: int
    pages: List[ParsedPage] = field(default_factory=list)


class BaseParser(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocumentBundle:
        """Parse the target document into a structured multimodal bundle."""
        pass
