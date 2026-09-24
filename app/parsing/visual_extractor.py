"""
Visual asset extraction engine for embedded images and rendered vector graphics.
"""
import logging
from pathlib import Path
from typing import List, Set
import pymupdf

from app.parsing.base import ExtractedVisualBlock
from app.schemas.document import BoundingBox
from app.storage.asset_store import AssetStore

logger = logging.getLogger("novacore.parsing.visual")

# Minimum visual thresholds to filter out bullets, icons, and 1px lines
MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64
MIN_IMAGE_PIXELS = 4096


class VisualExtractor:
    """Extracts raster images and renders vector graphic charts into high-res images."""

    def __init__(self, asset_store: AssetStore):
        self.asset_store = asset_store

    def extract_visuals_from_page(
        self,
        doc: pymupdf.Document,
        page: pymupdf.Page,
        page_number: int,
        doc_hash: str,
        seen_xrefs: Set[int],
    ) -> List[ExtractedVisualBlock]:
        """Extract all valid visual assets from a page and persist them."""
        extracted_visuals: List[ExtractedVisualBlock] = []
        visual_index = 1

        # 1. Extract embedded raster images
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue

            try:
                base_img = doc.extract_image(xref)
                if not base_img:
                    continue

                width = base_img.get("width", 0)
                height = base_img.get("height", 0)

                # Filter out tiny icons, decorative dividers, or tracking pixels
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT or (width * height) < MIN_IMAGE_PIXELS:
                    continue

                seen_xrefs.add(xref)
                image_bytes = base_img["image"]
                ext = base_img.get("ext", "png")

                # Get spatial location of the image on the page
                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    r = rects[0]
                    bbox = BoundingBox(
                        x0=float(r.x0),
                        y0=float(r.y0),
                        x1=float(r.x1),
                        y1=float(r.y1),
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                    )

                # Persist image deterministically in asset store
                saved_path = self.asset_store.save_image_bytes(
                    image_bytes=image_bytes,
                    doc_hash=doc_hash,
                    page_number=page_number,
                    asset_index=visual_index,
                    extension=ext,
                )

                extracted_visuals.append(
                    ExtractedVisualBlock(
                        image_path=saved_path,
                        page_number=page_number,
                        visual_index=visual_index,
                        bbox=bbox,
                        is_vector_graphic=False,
                    )
                )
                visual_index += 1

            except Exception as exc:
                logger.debug("Failed extracting raster image xref %d on page %d: %s", xref, page_number, exc)
                continue

        # 2. Vector Graphics Detection Fallback
        # If no raster images were found on the page, but complex vector drawings exist (charts/curves),
        # render the drawing area into a high-res image
        if not extracted_visuals:
            drawings = page.get_drawings()
            if len(drawings) >= 15:  # Significant drawing cluster indicative of a chart or diagram
                try:
                    # Union all drawing bounding boxes
                    rect_union = pymupdf.Rect()
                    for d in drawings:
                        rect_union |= d["rect"]

                    if (
                        rect_union.width > 120
                        and rect_union.height > 120
                        and not rect_union.is_empty
                        and not rect_union.is_infinite
                    ):
                        pix = page.get_pixmap(clip=rect_union, dpi=150)
                        saved_path = self.asset_store.save_image_bytes(
                            image_bytes=pix.tobytes("png"),
                            doc_hash=doc_hash,
                            page_number=page_number,
                            asset_index=visual_index,
                            extension="png",
                        )
                        bbox = BoundingBox(
                            x0=float(rect_union.x0),
                            y0=float(rect_union.y0),
                            x1=float(rect_union.x1),
                            y1=float(rect_union.y1),
                            page_width=float(page.rect.width),
                            page_height=float(page.rect.height),
                        )
                        extracted_visuals.append(
                            ExtractedVisualBlock(
                                image_path=saved_path,
                                page_number=page_number,
                                visual_index=visual_index,
                                bbox=bbox,
                                is_vector_graphic=True,
                            )
                        )
                except Exception as exc:
                    logger.debug("Vector graphics rendering bypassed on page %d: %s", page_number, exc)

        return extracted_visuals
