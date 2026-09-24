"""
Query understanding, intent classification, sub-query decomposition, and metadata filter extraction.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.config.constants import ModalityType
from app.schemas.query import FilterCriteria, ModalityIntent


@dataclass
class QueryAnalysisResult:
    """The structured output of upstream query analysis."""
    original_query: str
    intent: ModalityIntent
    sub_queries: List[str] = field(default_factory=list)
    extracted_filters: Optional[FilterCriteria] = None


class QueryAnalyzer:
    """Analyzes user questions to determine intent, sub-queries, and metadata constraints."""

    # Heuristic regex patterns for fast, sub-millisecond intent classification
    VISUAL_KEYWORDS = re.compile(r"\b(graph|chart|diagram|plot|trend|visual|flowchart|image|figure|picture)\b", re.IGNORECASE)
    TABULAR_KEYWORDS = re.compile(r"\b(table|breakdown|rows|columns|financials|metrics|comparison|yoy|quarterly)\b", re.IGNORECASE)
    PAGE_FILTER_PATTERN = re.compile(r"\bpages?\s*(\d+)(?:\s*(?:and|to|-)\s*(\d+))?\b", re.IGNORECASE)

    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """Analyze query string and return classification and decomposed queries."""
        clean_query = query.strip()

        # 1. Detect Intent
        has_visual = bool(self.VISUAL_KEYWORDS.search(clean_query))
        has_table = bool(self.TABULAR_KEYWORDS.search(clean_query))

        # Check for multi-part synthesis indicators
        has_numbering = bool(re.search(r"(?m)^\s*(?:\d+[\.\)]|\-|\*)\s+", clean_query)) or "include:" in clean_query.lower()

        if (has_visual and has_table) or has_numbering:
            intent = ModalityIntent.CROSS_MODAL
        elif has_visual:
            intent = ModalityIntent.VISUAL
        elif has_table:
            intent = ModalityIntent.TABULAR
        else:
            intent = ModalityIntent.TEXT_ONLY

        # 2. Extract Sub-Queries for Cross-Modal/Multi-Part Questions
        sub_queries = []
        if has_numbering:
            # Extract numbered points (e.g., "1. total revenue", "2. fastest growing region")
            lines = clean_query.split("\n")
            for line in lines:
                if re.match(r"^\s*(?:\d+[\.\)]|\-|\*)\s+", line):
                    cleaned_line = re.sub(r"^\s*(?:\d+[\.\)]|\-|\*)\s*", "", line).strip()
                    if cleaned_line:
                        sub_queries.append(cleaned_line)

        # Always include the original query as primary
        if not sub_queries:
            sub_queries = [clean_query]

        # 3. Extract Explicit Page Filters
        extracted_filters = None
        page_match = self.PAGE_FILTER_PATTERN.search(clean_query)
        if page_match:
            p1 = int(page_match.group(1))
            pages = [p1]
            if page_match.group(2):
                p2 = int(page_match.group(2))
                pages = list(range(min(p1, p2), max(p1, p2) + 1))
            extracted_filters = FilterCriteria(pages=pages)

        return QueryAnalysisResult(
            original_query=clean_query,
            intent=intent,
            sub_queries=sub_queries,
            extracted_filters=extracted_filters,
        )
