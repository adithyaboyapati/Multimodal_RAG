"""
Production asset store for managing multimodal artifacts, SHA-256 deduplication,
and Base64 Data URI serialization.
"""
import base64
from functools import lru_cache
import hashlib
import io
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from app.config.constants import DEFAULT_IMAGE_JPEG_QUALITY, DEFAULT_MAX_IMAGE_DIMENSION
from app.config.settings import Settings, get_settings


class AssetStore:
    """Manages persistent multimodal assets with collision-resistant content addressing."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_dir = self.settings.storage_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(file_path: Union[str, Path]) -> str:
        """Calculate the SHA-256 hash of a file for idempotent deduplication."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_document_asset_dir(self, doc_hash: str) -> Path:
        """Return and ensure directory for a specific document hash."""
        doc_dir = self.base_dir / doc_hash
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    def save_image_bytes(
        self,
        image_bytes: bytes,
        doc_hash: str,
        page_number: int,
        asset_index: int,
        extension: str = "png",
    ) -> Path:
        """Persist extracted image bytes with deterministic naming."""
        doc_dir = self.get_document_asset_dir(doc_hash)
        file_name = f"page_{page_number}_visual_{asset_index}.{extension}"
        dest_path = doc_dir / file_name
        dest_path.write_bytes(image_bytes)
        return dest_path

    @classmethod
    def image_to_data_uri(
        cls,
        image_path: Union[str, Path],
        max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
        quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
    ) -> str:
        """
        Convert image file to optimized Base64 JPEG Data URI for VLM inference and web rendering.
        Enforces maximum pixel dimension and quality compression. Results are cached in memory.
        """
        return cls._cached_image_to_data_uri(str(image_path), max_dimension, quality)

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_image_to_data_uri(
        image_path_str: str,
        max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
        quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
    ) -> str:
        path = Path(image_path_str)
        if not path.exists():
            raise FileNotFoundError(f"Visual asset not found at {path}")

        with Image.open(path) as img:
            # Convert RGBA/Palette/Grayscale to RGB for JPEG compatibility
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Constrain dimensions proportionally to control token consumption
            if img.width > max_dimension or img.height > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


def get_asset_store() -> AssetStore:
    """Dependency helper for asset store singleton."""
    return AssetStore()
