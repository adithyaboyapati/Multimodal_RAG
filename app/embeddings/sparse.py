"""
Production BM25 lexical sparse encoder for exact keyword and entity indexing.
Generates Pinecone-compatible sparse vectors {'indices': [...], 'values': [...]}.
"""
import re
import zlib
from collections import Counter
from typing import Any, Dict, List

from app.embeddings.base import BaseSparseEmbedder

# Matches alphanumeric terms, hyphenated codes (NC-942), and currencies ($55M)
TOKEN_PATTERN = re.compile(r"(?u)\b[\w\-\$]{2,}\b")
MAX_INDEX_RANGE = 2147483647  # Maximum positive 32-bit signed integer for Pinecone


class BM25SparseEmbedder(BaseSparseEmbedder):
    """
    Encodes text into sparse vector term-frequency distributions using BM25 weighting.
    Guarantees sorted, unique 32-bit indices required by Pinecone Serverless Hybrid Search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, avg_dl: float = 250.0):
        self.k1 = k1
        self.b = b
        self.avg_dl = avg_dl

    def _tokenize(self, text: str) -> List[str]:
        """Normalize text and extract search terms."""
        return TOKEN_PATTERN.findall(text.lower())

    def _hash_token(self, token: str) -> int:
        """Map string token deterministically to a 32-bit positive integer index."""
        return abs(zlib.crc32(token.encode("utf-8"))) % MAX_INDEX_RANGE

    def encode_document(self, text: str) -> Dict[str, Any]:
        """Encode a single document string into a BM25 sparse vector dictionary."""
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        dl = len(tokens)

        # Accumulate weights by token index to handle hash collisions gracefully
        index_weights: Dict[int, float] = {}
        for token, tf in counts.items():
            idx = self._hash_token(token)
            weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl)))
            index_weights[idx] = index_weights.get(idx, 0.0) + weight

        sorted_pairs = sorted(index_weights.items(), key=lambda x: x[0])
        return {
            "indices": [pair[0] for pair in sorted_pairs],
            "values": [round(pair[1], 4) for pair in sorted_pairs],
        }

    def encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encode a batch of document texts into sparse vectors."""
        return [self.encode_document(t) for t in texts]

    def encode_query(self, text: str) -> Dict[str, Any]:
        """Encode a query string into query-normalized BM25 weights."""
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        index_weights: Dict[int, float] = {}

        for token, tf in counts.items():
            idx = self._hash_token(token)
            # Query term weighting without document-length penalty
            weight = (tf * (self.k1 + 1.0)) / (tf + self.k1)
            index_weights[idx] = index_weights.get(idx, 0.0) + weight

        sorted_pairs = sorted(index_weights.items(), key=lambda x: x[0])
        return {
            "indices": [pair[0] for pair in sorted_pairs],
            "values": [round(pair[1], 4) for pair in sorted_pairs],
        }
