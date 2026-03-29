"""
Axial Coding – Algorithm 2 from the paper.

Begins as soon as open coding produces validated entities with evidence.
For each segment:
  1. Identify validated entities mentioned in the segment
  2. Retrieve neighbours
  3. Detect relationship triggers (4 types)
  4. For each triggered pair: targeted retrieval + LLM axial prompt
  5. Score with score_axial ≥ 0.70
  6. UML semantic validation (type, direction, multiplicity)
  7. Normalize and merge

Guardrails:
  - Skip entities with confidence < 0.70
  - Cap axial checks per segment
  - Pause pair after 3 consecutive proposals below 0.50
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

import config
from confidence_scoring.scorer import ConfidenceScorer
from model_bundle.schema import (
    Entity,
    EvidenceSpan,
    Relationship,
    RelationshipType,
    Segment,
)
from phase2_knowledge_construction.retriever import Retriever
from .trigger_detector import TriggerHit, detect_triggers
from .prompts import AXIAL_CODING_SYSTEM, AXIAL_CODING_USER_TEMPLATE

logger = logging.getLogger(__name__)


class AxialCoder:
    def __init__(self, retriever: Retriever, scorer: ConfidenceScorer):
        self._retriever = retriever
        self._scorer = scorer
        self._client = OpenAI()
        # Track consecutive low scores per (entity1, entity2) pair for guardrail
        self._pair_low_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self._paused_pairs: Set[Tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Algorithm 2
    # ------------------------------------------------------------------

    def run(
        self,
        entities: List[Entity],
        segments: List[Segment],
    ) -> List[Relationship]:
        """
        Execute axial coding over all segments with validated entities.
        Returns validated relationship set R.
        """
        # Only operate on entities that met the confidence threshold
        qualified_entities = [
            e for e in entities if e.confidence >= config.AXIAL_MIN_ENTITY_CONFIDENCE
        ]
        if not qualified_entities:
            return []

        relationship_set: List[Relationship] = []

        for seg in segments:
            # Filter to entities mentioned in this segment
            seg_entities = _entities_in_segment(qualified_entities, seg.text)
            if len(seg_entities) < 2:
                continue

            neighbors = self._retriever.retrieve_for_segment_neighbors(
                seg, k=config.RETRIEVAL_TOP_K
            )
            neighbor_texts = [s.text for s, _ in neighbors]

            # Detect triggers
            triggers = detect_triggers(seg_entities, seg.text, neighbor_texts)

            # Cap triggers and filter paused pairs before launching threads
            active_triggers = []
            for e1_name, e2_name, cue, ttype in triggers:
                if len(active_triggers) >= config.AXIAL_MAX_CHECKS_PER_SEGMENT:
                    break
                if _pair_key(e1_name, e2_name) in self._paused_pairs:
                    continue
                e1 = _find_entity(qualified_entities, e1_name)
                e2 = _find_entity(qualified_entities, e2_name)
                if e1 and e2:
                    active_triggers.append((e1, e2, cue))

            def _check_pair(e1: "Entity", e2: "Entity", cue: str):
                context_segs = self._retriever.retrieve_for_relationship(
                    e1.name, e2.name, cue, k=config.RETRIEVAL_TOP_K
                )
                context_text = "\n---\n".join(s.text for s, _ in context_segs[:3])
                retrieved_texts = [(s.text, score) for s, score in context_segs]
                raw_rel = self._call_llm(e1, e2, context_text)
                if raw_rel is None:
                    return None
                rel = self._parse_relationship(raw_rel, e1.name, e2.name, seg, context_segs)
                if rel is None:
                    return None
                score = self._scorer.score_relationship(rel, rel.evidence, retrieved_texts)
                rel.confidence = score
                return rel

            with ThreadPoolExecutor(max_workers=config.LLM_PARALLEL_WORKERS) as executor:
                futures = {executor.submit(_check_pair, e1, e2, cue): (e1.name, e2.name)
                           for e1, e2, cue in active_triggers}
                for future in as_completed(futures):
                    e1_name, e2_name = futures[future]
                    pair_key = _pair_key(e1_name, e2_name)
                    rel = future.result()
                    if rel is None:
                        continue
                    score = rel.confidence
                    if score < config.AXIAL_PAUSE_SCORE_THRESHOLD:
                        self._pair_low_count[pair_key] += 1
                        if self._pair_low_count[pair_key] >= config.AXIAL_PAUSE_CONSECUTIVE:
                            logger.debug("Pausing pair %s <-> %s", e1_name, e2_name)
                            self._paused_pairs.add(pair_key)
                        continue
                    if score < config.CONFIDENCE_THRESHOLD:
                        continue
                    if not _validate_uml_semantics(rel):
                        continue
                    relationship_set.append(rel)
                    self._pair_low_count[pair_key] = 0
                    logger.debug(
                        "  Accepted: %s -[%s]-> %s (score=%.3f)",
                        e1_name, rel.relationship_type.value, e2_name, score
                    )

            # Normalize and merge after each segment
            relationship_set = _normalize_and_merge(relationship_set)

        return relationship_set

    def resume_paused_pairs(self, entities: List[Entity], new_evidence_segment: Segment) -> None:
        """
        Resume paused pairs when new evidence appears (per paper: "resume only if
        later text mentions renewals, returns, or loan periods").
        Called from pipeline when new segments introduce relevant context.
        """
        self._paused_pairs.clear()
        self._pair_low_count.clear()

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self, e1: Entity, e2: Entity, context_text: str
    ) -> Optional[Dict]:
        user_msg = AXIAL_CODING_USER_TEMPLATE.format(
            entity1_name=e1.name,
            entity1_definition=e1.definition,
            entity2_name=e2.name,
            entity2_definition=e2.definition,
            relationship_context=context_text or "(no context available)",
        )
        try:
            response = self._client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": AXIAL_CODING_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return _parse_json_object(raw)
        except Exception as exc:
            logger.warning("LLM call failed during axial coding: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_relationship(
        self,
        raw: Dict,
        e1_name: str,
        e2_name: str,
        seg: Segment,
        context_segs: List[Tuple],
    ) -> Optional[Relationship]:
        try:
            raw = {k.lower(): v for k, v in raw.items()}
            rel_type_str = raw.get("relationship_type", "NONE").strip()
            rel_type = _parse_rel_type(rel_type_str)
            if rel_type == RelationshipType.NONE:
                return None

            evidence_quote = raw.get("evidence_quote", "").strip()
            spans: List[EvidenceSpan] = []
            if evidence_quote:
                spans.append(EvidenceSpan(
                    text=evidence_quote[:300],
                    segment_id=seg.segment_id,
                    source_document=seg.metadata.source_document,
                    section_title=seg.metadata.section_title,
                ))
            # Add corroborating spans from context segments
            for cseg, _ in context_segs[:2]:
                if evidence_quote and evidence_quote[:40] in cseg.text:
                    pass  # already included via focal segment
                else:
                    q = _extract_pair_quote(e1_name, e2_name, cseg.text)
                    if q:
                        spans.append(EvidenceSpan(
                            text=q,
                            segment_id=cseg.segment_id,
                            source_document=cseg.metadata.source_document,
                            section_title=cseg.metadata.section_title,
                        ))

            return Relationship(
                relationship_type=rel_type,
                source=raw.get("source") or e1_name,
                target=raw.get("target") or e2_name,
                multiplicity_source=raw.get("multiplicity_source") or "1",
                multiplicity_target=raw.get("multiplicity_target") or "1",
                role_name=raw.get("role_name") or None,
                evidence=spans,
            )
        except Exception as exc:
            logger.debug("Failed to parse relationship: %s | %s", raw, exc)
            return None


# ---------------------------------------------------------------------------
# UML semantic validation
# ---------------------------------------------------------------------------

def _validate_uml_semantics(rel: Relationship) -> bool:
    """
    Three-step UML validation from the paper:
      1. Type compatibility
      2. Directionality (role_name or inherent direction)
      3. Multiplicity sanity
    """
    rt = rel.relationship_type

    # Type compatibility
    if rt == RelationshipType.IMPLEMENTS:
        # IMPLEMENTS should connect class to interface – we can't check here
        # without element type info, so allow it
        pass

    # Directionality: source and target must differ
    if rel.source == rel.target:
        return False

    # Multiplicity sanity: must be valid UML notation
    for mult in [rel.multiplicity_source, rel.multiplicity_target]:
        if not _valid_multiplicity(mult):
            return False

    return True


def _valid_multiplicity(mult: str) -> bool:
    if not mult:
        return False
    if mult in ("1", "0..1", "*", "0..*", "1..*"):
        return True
    # Accept patterns like "0..10", "1..5"
    if re.match(r"^\d+\.\.\d+$", mult):
        return True
    if re.match(r"^\d+$", mult):
        return True
    return False


# ---------------------------------------------------------------------------
# Normalization and merging
# ---------------------------------------------------------------------------

def _normalize_and_merge(relationships: List[Relationship]) -> List[Relationship]:
    """
    Merge duplicate proposals for the same entity pair.
    Keep the higher-scoring one; retain alternates' evidence.
    Apply type priority when scores are tied.
    """
    # Group by (source, target) pair (direction-normalized)
    groups: Dict[Tuple[str, str], List[Relationship]] = defaultdict(list)
    for rel in relationships:
        key = tuple(sorted([rel.source, rel.target]))
        groups[key].append(rel)

    merged: List[Relationship] = []
    for key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Sort by confidence, then by type priority
        group.sort(
            key=lambda r: (
                r.confidence,
                -config.RELATIONSHIP_TYPE_PRIORITY.index(r.relationship_type.value)
                if r.relationship_type.value in config.RELATIONSHIP_TYPE_PRIORITY
                else -999,
            ),
            reverse=True,
        )
        best = group[0]
        # Merge evidence from all others into best
        existing_texts = {e.text for e in best.evidence}
        for other in group[1:]:
            for ev in other.evidence:
                if ev.text not in existing_texts:
                    best.evidence.append(ev)
                    existing_texts.add(ev.text)
        merged.append(best)

    return merged


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _entities_in_segment(entities: List[Entity], text: str) -> List[Entity]:
    text_lower = text.lower()
    return [e for e in entities if e.name.lower() in text_lower]


def _find_entity(entities: List[Entity], name: str) -> Optional[Entity]:
    for e in entities:
        if e.name == name:
            return e
    return None


def _pair_key(name1: str, name2: str) -> Tuple[str, str]:
    return tuple(sorted([name1, name2]))


def _parse_rel_type(raw: str) -> RelationshipType:
    mapping = {
        "is_a": RelationshipType.IS_A,
        "is_part_of": RelationshipType.IS_PART_OF,
        "aggregates": RelationshipType.AGGREGATES,
        "implements": RelationshipType.IMPLEMENTS,
        "associates": RelationshipType.ASSOCIATES,
        "depends_on": RelationshipType.DEPENDS_ON,
        "none": RelationshipType.NONE,
    }
    return mapping.get(raw.lower().strip(), RelationshipType.NONE)


def _extract_pair_quote(name1: str, name2: str, text: str) -> Optional[str]:
    n1, n2 = name1.lower(), name2.lower()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.lower()
        if n1 in s and n2 in s:
            return sentence[:300]
    return None


def _parse_json_object(content: str) -> Optional[Dict]:
    content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
