"""
Open Coding – Algorithm 1 from the paper.

For each segment:
  1. Retrieve top-k neighbours from Phase II vector store
  2. Call LLM with open coding prompt (Listing 2)
  3. Validate each candidate with evidence score (score_open ≥ 0.70)
  4. Verify textual presence (names appear verbatim in requirements)
  5. Merge duplicates and variants after each segment

LLM proposes candidates only.
The ConfidenceScorer validates independently.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

import config
from confidence_scoring.scorer import ConfidenceScorer
from model_bundle.schema import (
    Attribute,
    ElementType,
    Entity,
    EvidenceSpan,
    Operation,
    Segment,
)
from phase2_knowledge_construction.retriever import Retriever
from .merger import merge_duplicate_entities
from .prompts import OPEN_CODING_SYSTEM, OPEN_CODING_USER_TEMPLATE

logger = logging.getLogger(__name__)


class OpenCoder:
    def __init__(self, retriever: Retriever, scorer: ConfidenceScorer):
        self._retriever = retriever
        self._scorer = scorer
        self._client = OpenAI()

    # ------------------------------------------------------------------
    # Algorithm 1: Open Coding
    # ------------------------------------------------------------------

    def run(self, segments: List[Segment]) -> List[Entity]:
        """
        Execute open coding over all segments in parallel.
        Returns validated and merged entity set E.
        """
        # Process segments in parallel batches; merge is done after all batches
        results: List[List[Entity]] = [[] for _ in segments]

        def _process_segment(idx: int, seg: Segment) -> tuple:
            logger.debug("Open coding segment: %s", seg.segment_id)
            neighbors = self._retriever.retrieve_for_segment_neighbors(
                seg, k=config.RETRIEVAL_TOP_K
            )
            retrieved_texts = [(s.text, score) for s, score in neighbors]
            context_text = "\n---\n".join(s.text for s, _ in neighbors[:3])
            candidates = self._call_llm(seg.text, context_text)
            accepted = []
            for raw in candidates:
                entity = self._parse_entity(raw, seg)
                if entity is None:
                    continue
                evidence_spans = self._build_evidence_spans(entity, seg, neighbors)
                entity.evidence = evidence_spans
                score = self._scorer.score_entity(entity, evidence_spans, retrieved_texts)
                entity.confidence = score
                if score >= config.CONFIDENCE_THRESHOLD:
                    accepted.append(entity)
                    logger.debug("  Accepted: %s (score=%.3f)", entity.name, score)
                elif config.OPEN_CODING_REQUERY_ON_LOW_CONFIDENCE:
                    entity = self._requery_and_rescore(entity, seg, score)
                    if entity and entity.confidence >= config.CONFIDENCE_THRESHOLD:
                        accepted.append(entity)
                        logger.debug("  Accepted after requery: %s (score=%.3f)", entity.name, entity.confidence)
            return idx, accepted

        with ThreadPoolExecutor(max_workers=config.LLM_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_process_segment, i, seg): i for i, seg in enumerate(segments)}
            for future in as_completed(futures):
                idx, accepted = future.result()
                results[idx] = accepted

        # Flatten in segment order then merge duplicates once
        entity_set: List[Entity] = []
        for seg_entities in results:
            entity_set.extend(seg_entities)
        entity_set = merge_duplicate_entities(entity_set)
        return entity_set

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, segment_text: str, context_text: str) -> List[Dict]:
        """Call LLM with open coding prompt. Returns list of raw entity dicts."""
        user_msg = OPEN_CODING_USER_TEMPLATE.format(
            segment_text=segment_text,
            retrieved_context=context_text or "(no additional context)",
        )
        try:
            response = self._client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": OPEN_CODING_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw_content = response.choices[0].message.content.strip()
            return _parse_json_array(raw_content)
        except Exception as exc:
            logger.warning("LLM call failed during open coding: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Parsing LLM output into Entity objects
    # ------------------------------------------------------------------

    def _parse_entity(self, raw: Dict, seg: Segment) -> Optional[Entity]:
        """Convert raw LLM dict to Entity. Returns None if malformed."""
        try:
            # Normalize keys to lowercase to handle models that return uppercase keys
            raw = {k.lower(): v for k, v in raw.items()}
            name = raw.get("name", "").strip()
            if not name:
                return None

            # Verify name appears in segment text (verbatim presence check)
            if name.lower() not in seg.text.lower():
                # Try partial match for compound names
                first_word = name.split()[0] if " " in name else name
                if first_word.lower() not in seg.text.lower():
                    return None

            element_type = _parse_element_type(raw.get("element_type", "Class"))

            attributes = []
            for a in raw.get("attributes", []):
                attr_name = a.get("name", "").strip()
                if attr_name:
                    attributes.append(Attribute(
                        name=attr_name,
                        type=a.get("type", "String"),
                        owner=name,
                    ))

            operations = []
            for op in raw.get("operations", []):
                op_name = op.get("name", "").strip()
                if op_name:
                    operations.append(Operation(
                        name=op_name,
                        parameters=op.get("parameters", []),
                        return_type=op.get("return_type", "void"),
                        owner=name,
                    ))

            return Entity(
                name=name,
                definition=raw.get("definition", ""),
                element_type=element_type,
                attributes=attributes,
                operations=operations,
            )
        except Exception as exc:
            logger.debug("Failed to parse entity from LLM output: %s | %s", raw, exc)
            return None

    # ------------------------------------------------------------------
    # Evidence span construction
    # ------------------------------------------------------------------

    def _build_evidence_spans(
        self,
        entity: Entity,
        seg: Segment,
        neighbors: List[Tuple],
    ) -> List[EvidenceSpan]:
        """Build evidence spans by finding verbatim quotes in segment and neighbors."""
        spans: List[EvidenceSpan] = []

        # Primary: find the entity name in the focal segment
        quote = _extract_sentence_containing(entity.name, seg.text)
        if quote:
            spans.append(EvidenceSpan(
                text=quote,
                segment_id=seg.segment_id,
                source_document=seg.metadata.source_document,
                section_title=seg.metadata.section_title,
                page_number=seg.metadata.page_number,
            ))

        # Secondary: corroborating quotes from neighbours
        for neighbor_seg, _ in neighbors[:3]:
            nquote = _extract_sentence_containing(entity.name, neighbor_seg.text)
            if nquote and nquote != quote:
                spans.append(EvidenceSpan(
                    text=nquote,
                    segment_id=neighbor_seg.segment_id,
                    source_document=neighbor_seg.metadata.source_document,
                    section_title=neighbor_seg.metadata.section_title,
                    page_number=neighbor_seg.metadata.page_number,
                ))

        return spans

    # ------------------------------------------------------------------
    # Re-query on low confidence
    # ------------------------------------------------------------------

    def _requery_and_rescore(
        self, entity: Entity, seg: Segment, prior_score: float
    ) -> Optional[Entity]:
        """
        Re-query the vector store with refined terms and perform a second pass.
        Returns updated entity if score improves above threshold, else None.
        """
        refined_neighbors = self._retriever.retrieve_for_entity(
            entity_name=entity.name,
            context_terms=[a.name for a in entity.attributes[:3]],
        )
        refined_texts = [(s.text, score) for s, score in refined_neighbors]

        # Rebuild evidence from refined context
        new_spans = self._build_evidence_spans(entity, seg, refined_neighbors)
        entity.evidence = new_spans

        new_score = self._scorer.score_entity(entity, new_spans, refined_texts)
        entity.confidence = new_score
        return entity if new_score >= config.CONFIDENCE_THRESHOLD else None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_element_type(raw: str) -> ElementType:
    mapping = {
        "class": ElementType.CLASS,
        "interface": ElementType.INTERFACE,
        "enumeration": ElementType.ENUMERATION,
        "enum": ElementType.ENUMERATION,
        "actor": ElementType.ACTOR,
    }
    return mapping.get(raw.lower().strip(), ElementType.CLASS)


def _extract_sentence_containing(term: str, text: str) -> Optional[str]:
    """Extract the sentence (or phrase up to 200 chars) that contains the term."""
    if not term or not text:
        return None
    term_lower = term.lower()
    text_lower = text.lower()
    idx = text_lower.find(term_lower)
    if idx == -1:
        return None
    # Walk back to sentence start
    start = text.rfind(".", 0, idx)
    start = (start + 1) if start >= 0 else 0
    # Walk forward to sentence end
    end = text.find(".", idx)
    end = (end + 1) if end >= 0 else len(text)
    sentence = text[start:end].strip()
    return sentence[:300] if sentence else None


def _parse_json_array(content: str) -> List[Dict]:
    """Extract and parse a JSON array from LLM response (handles markdown fences)."""
    # Strip markdown code fences if present
    content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
    # Find the first [ ... ] block
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        logger.debug("Failed to parse JSON from LLM response: %s", content[:200])
        return []
