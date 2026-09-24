"""
Table extraction and structured serialization engine.
"""
import logging
from typing import List
import pandas as pd
import pymupdf

from app.parsing.base import ExtractedTableBlock
from app.schemas.document import BoundingBox

logger = logging.getLogger("novacore.parsing.table")


class TableExtractor:
    """Detects and serializes tabular structures into clean Markdown tables."""

    @staticmethod
    def extract_tables_from_page(page: pymupdf.Page, page_number: int) -> List[ExtractedTableBlock]:
        """Detect tables on a PyMuPDF page and convert to markdown blocks with bounding boxes."""
        extracted_tables = []
        try:
            detected_tables = page.find_tables().tables
        except Exception as exc:
            logger.debug("Table detection bypassed on page %d: %s", page_number, exc)
            return extracted_tables

        for idx, table in enumerate(detected_tables, start=1):
            try:
                df = table.to_pandas()
                if df.empty:
                    continue

                # Clean column headers and fill nulls for clean LLM parsing
                df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
                df = df.dropna(how="all")
                df = df.fillna("")

                markdown_str = df.to_markdown(index=False)
                if not markdown_str or not markdown_str.strip():
                    continue

                bbox = BoundingBox(
                    x0=float(table.bbox[0]),
                    y0=float(table.bbox[1]),
                    x1=float(table.bbox[2]),
                    y1=float(table.bbox[3]),
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                )

                extracted_tables.append(
                    ExtractedTableBlock(
                        markdown=markdown_str.strip(),
                        page_number=page_number,
                        table_number=idx,
                        bbox=bbox,
                    )
                )
            except Exception as exc:
                logger.debug("Failed extracting table %d on page %d: %s", idx, page_number, exc)
                continue

        return extracted_tables
