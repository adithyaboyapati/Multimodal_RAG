"""
Visual summarization engine converting images into high-density semantic text chunks
using Groq Vision models with disk caching and rate-limit retries.
"""
import time
from pathlib import Path
from typing import Optional

from groq import Groq

from app.config.constants import PromptTemplates
from app.config.settings import Settings, get_settings
from app.parsing.base import ExtractedVisualBlock
from app.storage.asset_store import AssetStore, get_asset_store


class VisualSummarizer:
    """Generates structured, searchable text descriptions of figures and charts."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        asset_store: Optional[AssetStore] = None,
        client: Optional[Groq] = None,
    ):
        self.settings = settings or get_settings()
        self.asset_store = asset_store or get_asset_store()
        self.client = client or Groq(api_key=self.settings.groq_api_key)

    def summarize_visual(
        self,
        visual: ExtractedVisualBlock,
        document_name: str,
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> str:
        """
        Summarize a visual block into a factual text chunk.
        Uses persistent disk cache to avoid redundant API fees on re-ingestion.
        """
        cache_file = visual.image_path.with_suffix(".summary.txt")

        # 1. Return from disk cache if present
        if use_cache and cache_file.exists():
            cached_text = cache_file.read_text(encoding="utf-8").strip()
            if cached_text:
                return cached_text

        # 2. Serialize image to Base64 JPEG
        data_uri = self.asset_store.image_to_data_uri(visual.image_path)

        prompt = PromptTemplates.VISUAL_SUMMARIZER.format(
            page_number=visual.page_number,
            document_name=document_name,
        ).strip()

        # 3. Call Groq VLM with exponential backoff
        delay = 1.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        }
                    ],
                    temperature=0.0,
                    max_completion_tokens=600,
                )
                summary = response.choices[0].message.content.strip()

                # Persist to disk cache
                if use_cache and summary:
                    cache_file.write_text(summary, encoding="utf-8")

                return summary

            except Exception as e:
                last_error = e
                # Check for rate-limiting
                time.sleep(delay)
                delay *= 2

        fallback_summary = f"[Visual element extracted from Page {visual.page_number} of {document_name}. Summary generation failed: {last_error}]"
        return fallback_summary
