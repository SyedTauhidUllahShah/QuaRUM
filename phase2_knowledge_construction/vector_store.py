"""
Phase II – Vector store.

Stores embeddings and metadata in FAISS with a Hierarchical Navigable
Small World (HNSW) index.  The database persists vectors, supports
approximate nearest-neighbour search, and exposes metadata filters.

HNSW is chosen for superior efficiency in approximate nearest-neighbour
search, enabling scalable retrieval across large document collections.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

import config
from model_bundle.schema import Segment, SegmentMetadata


class VectorStore:
    """
    FAISS HNSW index paired with an in-memory metadata store.

    Segment IDs are stored in a parallel list so that FAISS integer
    indices map back to segment IDs and their metadata.
    """

    # HNSW construction parameters (standard defaults)
    _HNSW_M = 32               # number of connections per element
    _HNSW_EF_CONSTRUCTION = 200

    def __init__(self):
        self._index: Optional[faiss.IndexHNSWFlat] = None
        self._segment_ids: List[str] = []                       # position -> segment_id
        self._segments: Dict[str, Segment] = {}                 # segment_id -> Segment
        self._dim: int = config.EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, segments: List[Segment], embeddings: np.ndarray) -> None:
        """
        Build the HNSW index from segments and their pre-computed embeddings.

        embeddings: shape (len(segments), EMBEDDING_DIM), float32, L2-normalised.
        """
        assert len(segments) == len(embeddings), "segments and embeddings must have equal length"
        assert embeddings.shape[1] == self._dim

        # FAISS IndexHNSWFlat uses inner product for cosine sim on normalised vectors
        self._index = faiss.IndexHNSWFlat(self._dim, self._HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self._HNSW_EF_CONSTRUCTION
        self._index.hnsw.efSearch = 64    # runtime search parameter

        self._segment_ids = []
        self._segments = {}

        for seg, emb in zip(segments, embeddings):
            self._segment_ids.append(seg.segment_id)
            self._segments[seg.segment_id] = seg

        vecs = embeddings.astype(np.float32)
        self._index.add(vecs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump({
                "segment_ids": self._segment_ids,
                "segments": self._segments,
                "dim": self._dim,
            }, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        vs = cls()
        vs._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        vs._segment_ids = meta["segment_ids"]
        vs._segments = meta["segments"]
        vs._dim = meta["dim"]
        return vs

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int = config.RETRIEVAL_TOP_K,
        section_filter: Optional[str] = None,
    ) -> List[Tuple[Segment, float]]:
        """
        Return top-k (Segment, score) pairs ranked by cosine similarity.

        section_filter: if provided, only return segments whose section_title
        contains this string (case-insensitive). Retrieves 3*k candidates
        before filtering to ensure k results when possible.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        fetch_k = k * 3 if section_filter else k
        fetch_k = min(fetch_k, self._index.ntotal)

        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, fetch_k)

        results: List[Tuple[Segment, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            seg_id = self._segment_ids[idx]
            seg = self._segments[seg_id]
            if section_filter:
                title = seg.metadata.section_title or ""
                if section_filter.lower() not in title.lower():
                    continue
            results.append((seg, float(score)))
            if len(results) >= k:
                break

        return results

    def get_segment(self, segment_id: str) -> Optional[Segment]:
        return self._segments.get(segment_id)

    @property
    def size(self) -> int:
        return len(self._segment_ids)
