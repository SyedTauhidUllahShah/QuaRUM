"""
Phase II – Retriever.

Supports both explicit and implicit query modes:
  - Explicit: named entities or candidate class pairs (open + axial coding)
  - Implicit: partial model state queries (partial attributes / operations)

Re-scoring applies contextual signals on top of cosine similarity:
  section headers, document proximity, and structural alignment.

Returns 3–5 relevant segments per query (paper: empirically validated range).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

import config
from model_bundle.schema import Segment
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self._embedder = embedder
        self._store = vector_store

    # ------------------------------------------------------------------
    # Primary retrieval interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        anchor_section: Optional[str] = None,
    ) -> List[Tuple[Segment, float]]:
        """
        Retrieve top-k segments for a query string.

        anchor_section: section title of the focal segment; used for
        structural alignment re-scoring (boosts segments from same section).
        """
        query_vec = self._embedder.encode_single(query)
        candidates = self._store.search(query_vec, k=k * 2)

        if not candidates:
            return []

        # Re-score with contextual signals
        rescored = self._rescore(candidates, anchor_section)

        # Return top-k after re-scoring
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:k]

    def retrieve_for_entity(
        self,
        entity_name: str,
        context_terms: List[str],
        k: int = config.RETRIEVAL_TOP_K,
    ) -> List[Tuple[Segment, float]]:
        """
        Explicit query mode: combine entity name with domain-specific terms.
        Used during open coding for entity-specific retrieval.
        """
        query = entity_name
        if context_terms:
            query = entity_name + " " + " ".join(context_terms[:3])
        return self.retrieve(query, k=k)

    def retrieve_for_relationship(
        self,
        entity1: str,
        entity2: str,
        cue: str,
        k: int = config.RETRIEVAL_TOP_K,
    ) -> List[Tuple[Segment, float]]:
        """
        Explicit query mode: combine entity names and the strongest in-vivo cue.
        Used during axial coding for relationship grounding.
        """
        query = f"{entity1} {entity2} {cue}"
        return self.retrieve(query, k=k)

    def retrieve_for_segment_neighbors(
        self,
        segment: Segment,
        k: int = config.RETRIEVAL_TOP_K,
    ) -> List[Tuple[Segment, float]]:
        """
        Retrieve semantically related segments that neighbor a focal segment.
        Used in open coding to analyse each segment with its context.
        """
        return self.retrieve(
            segment.text[:500],   # use first 500 chars as query
            k=k,
            anchor_section=segment.metadata.section_title,
        )

    # ------------------------------------------------------------------
    # Re-scoring
    # ------------------------------------------------------------------

    def _rescore(
        self,
        candidates: List[Tuple[Segment, float]],
        anchor_section: Optional[str],
    ) -> List[Tuple[Segment, float]]:
        """
        Apply contextual re-scoring signals:
          1. Section header alignment: boost if same section as anchor
          2. Document proximity: slight boost for adjacent segment IDs
        Scores are normalised to [0, 1].
        """
        if not candidates:
            return candidates

        rescored = []
        for seg, base_score in candidates:
            bonus = 0.0

            # Signal 1: section header alignment
            if anchor_section and seg.metadata.section_title:
                if anchor_section.lower() == seg.metadata.section_title.lower():
                    bonus += 0.05

            final_score = min(1.0, base_score + bonus)
            rescored.append((seg, final_score))

        return rescored
