"""
Phase II – Embedding.

Uses sentence-transformers all-MiniLM-L6-v2 (dimension=384).
Model is NOT fine-tuned (per paper).
Maps segments to high-dimensional dense vectors that capture semantic meaning,
placing semantically similar segments near each other in vector space.
"""

from __future__ import annotations

from typing import List

import numpy as np

import config


class Embedder:
    """Wraps all-MiniLM-L6-v2 for segment encoding."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.
        Returns array of shape (len(texts), EMBEDDING_DIM).
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product after normalisation
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns shape (EMBEDDING_DIM,)."""
        return self.encode([text])[0]
