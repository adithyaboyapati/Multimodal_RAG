"""
Production layout-aware PDF parser with spatial table filtering and visual extraction.
"""
from pathlib import Path
from typing import Optional, Set
import pymupdf

from app.parsing.base import (
    BaseParser,
    ExtractedTextBlock,
    ParsedDocumentBundle,
    ParsedPage,
)
from app.parsing.table_extractor import TableExtractor
from app.parsing.visual_extractor import VisualExtractor
from app.schemas.document import BoundingBox
from app.storage.asset_store import AssetStore, get_asset_store

SCANNED_PAGE_CHAR_THRESHOLD = 50


class PDFParser(BaseParser):
    """Layout-aware PDF parsing engine isolating text, tables, and visuals."""

    def __init__(self, asset_store: Optional[AssetStore] = None):
        self.asset_store = asset_store or get_asset_store()
        self.table_extractor = TableExtractor()
        self.visual_extractor = VisualExtractor(self.asset_store)

    def parse(self, file_path: Path) -> ParsedDocumentBundle:
        """Parse a PDF document into structured multimodal pages."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        doc_hash = self.asset_store.compute_file_hash(path)
        doc = pymupdf.open(str(path))

        parsed_pages = []
        seen_xrefs: Set[int] = set()

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_number = page_index + 1

                # 1. Structured Table Extraction
                tables = self.table_extractor.extract_tables_from_page(page, page_number)
                table_rects = [
                    pymupdf.Rect(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1)
                    for t in tables
                ]

                # 2. Text Extraction with Spatial Table Filtering
                # Prevents raw table text from contaminating text chunks
                blocks = page.get_text("blocks")
                text_blocks = []
                total_chars = 0

                for b in blocks:
                    # b = (x0, y0, x1, y1, text, block_no, block_type)
                    # block_type 0 = text, 1 = image
                    if b[6] == 0:
                        raw_text = b[4].strip()
                        if not raw_text:
                            continue

                        block_rect = pymupdf.Rect(b[0], b[1], b[2], b[3])

                        # Check if block falls inside any detected table
                        is_inside_table = False
                        for tr in table_rects:
                            if block_rect.intersects(tr):
                                intersection_area = (block_rect & tr).get_area()
                                block_area = block_rect.get_area()
                                if block_area > 0 and (intersection_area / block_area) > 0.4:
                                    is_inside_table = True
                                    break

                        if not is_inside_table:
                            bbox = BoundingBox(
                                x0=float(b[0]),
                                y0=float(b[1]),
                                x1=float(b[2]),
                                y1=float(b[3]),
                                page_width=float(page.rect.width),
                                page_height=float(page.rect.height),
                            )
                            text_blocks.append(
                                ExtractedTextBlock(
                                    text=raw_text,
                                    page_number=page_number,
                                    bbox=bbox,
                                )
                            )
                            total_chars += len(raw_text)

                # 3. Visual Assets Extraction (Raster + Vector)
                visuals = self.visual_extractor.extract_visuals_from_page(
                    doc=doc,
                    page=page,
                    page_number=page_number,
                    doc_hash=doc_hash,
                    seen_xrefs=seen_xrefs,
                )

                # 4. Scanned Page Heuristic
                is_scanned = (total_chars < SCANNED_PAGE_CHAR_THRESHOLD and len(visuals) > 0 and len(tables) == 0)

                parsed_pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        text_blocks=text_blocks,
                        tables=tables,
                        visuals=visuals,
                        is_scanned=is_scanned,
                        raw_character_count=total_chars,
                    )
                )
        finally:
            doc.close()

        return ParsedDocumentBundle(
            source_name=path.name,
            document_hash=doc_hash,
            total_pages=len(parsed_pages),
            pages=parsed_pages,
        )
