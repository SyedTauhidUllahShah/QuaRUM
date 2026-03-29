"""
Open Coding – Duplicate/variant merging.

Merges entities that refer to the same concept using:
  - Name similarity (exact match or cosine similarity ≥ 0.85)
  - Attribute overlap
  - Contextual fit (shared evidence segment IDs)

When merging: keep the highest-confidence representation,
carry forward the full set of supporting evidence, and record
variant names as aliases.

Paper example: "Patron" and "LibraryUser" merge when attributes
and surrounding context align.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from model_bundle.schema import Attribute, Entity, EvidenceSpan, Operation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_duplicate_entities(entities: List[Entity]) -> List[Entity]:
    """
    Identify and merge duplicate/variant entities.
    Returns a deduplicated list.
    """
    if len(entities) <= 1:
        return entities

    merged: List[Entity] = []
    used: Set[int] = set()

    for i, entity_a in enumerate(entities):
        if i in used:
            continue
        cluster = [entity_a]
        cluster_indices = [i]

        for j, entity_b in enumerate(entities):
            if j <= i or j in used:
                continue
            if _should_merge(entity_a, entity_b):
                cluster.append(entity_b)
                cluster_indices.append(j)
                used.add(j)

        used.add(i)
        merged.append(_merge_cluster(cluster))

    return merged


# ---------------------------------------------------------------------------
# Merge decision
# ---------------------------------------------------------------------------

def _should_merge(a: Entity, b: Entity) -> bool:
    """
    Return True if entities a and b refer to the same concept.
    Checks: name similarity ≥ 0.85, OR attribute overlap ≥ 0.5, OR
    shared evidence segments.
    """
    name_sim = _name_similarity(a.name, b.name)
    if name_sim >= 0.85:
        return True

    # Check shared evidence segment IDs
    segs_a = {e.segment_id for e in a.evidence}
    segs_b = {e.segment_id for e in b.evidence}
    if segs_a and segs_b and len(segs_a & segs_b) / max(len(segs_a), len(segs_b)) >= 0.5:
        attr_overlap = _attribute_overlap(a, b)
        if attr_overlap >= 0.5:
            return True

    return False


def _name_similarity(name_a: str, name_b: str) -> float:
    """
    Compute character-level name similarity using normalised edit distance.
    Also checks exact match after normalisation.
    """
    a = _normalise_name(name_a)
    b = _normalise_name(name_b)

    if a == b:
        return 1.0

    # Levenshtein-based similarity
    dist = _levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len


def _attribute_overlap(a: Entity, b: Entity) -> float:
    """Fraction of attribute names shared between two entities."""
    names_a = {_normalise_name(attr.name) for attr in a.attributes}
    names_b = {_normalise_name(attr.name) for attr in b.attributes}
    if not names_a or not names_b:
        return 0.0
    intersection = names_a & names_b
    union = names_a | names_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Merge execution
# ---------------------------------------------------------------------------

def _merge_cluster(cluster: List[Entity]) -> Entity:
    """
    Merge a cluster of entities into one.
    Keep the entity with the highest confidence as the base.
    Carry forward all evidence and record variant names as aliases.
    """
    cluster.sort(key=lambda e: e.confidence, reverse=True)
    base = cluster[0]

    all_aliases: List[str] = list(base.aliases)
    all_evidence: List[EvidenceSpan] = list(base.evidence)
    all_attributes: Dict[str, Attribute] = {a.name: a for a in base.attributes}
    all_operations: Dict[str, Operation] = {op.name: op for op in base.operations}

    for other in cluster[1:]:
        # Record variant name as alias
        if other.name != base.name and other.name not in all_aliases:
            all_aliases.append(other.name)
        all_aliases.extend(a for a in other.aliases if a not in all_aliases)

        # Merge evidence (deduplicate by text)
        existing_texts = {e.text for e in all_evidence}
        for ev in other.evidence:
            if ev.text not in existing_texts:
                all_evidence.append(ev)
                existing_texts.add(ev.text)

        # Merge attributes: keep higher-confidence version
        for attr in other.attributes:
            # Find if a similar attribute already exists (by name similarity/containment)
            existing_key = _find_similar_attr_key(attr.name, all_attributes)
            if existing_key is None:
                all_attributes[attr.name] = attr
            elif attr.confidence > all_attributes[existing_key].confidence:
                all_attributes[existing_key] = attr
            else:
                # Merge evidence into existing
                existing_attr = all_attributes[existing_key]
                existing_ev_texts = {e.text for e in existing_attr.evidence}
                for ev in attr.evidence:
                    if ev.text not in existing_ev_texts:
                        existing_attr.evidence.append(ev)

        # Merge operations: keep higher-confidence version
        for op in other.operations:
            if op.name not in all_operations:
                all_operations[op.name] = op
            elif op.confidence > all_operations[op.name].confidence:
                all_operations[op.name] = op

    base.aliases = all_aliases
    base.evidence = all_evidence
    base.attributes = _dedup_attributes(list(all_attributes.values()))
    base.operations = list(all_operations.values())
    return base


def _dedup_attributes(attrs: List[Attribute]) -> List[Attribute]:
    """Deduplicate attributes within a single entity using containment/similarity."""
    result: Dict[str, Attribute] = {}
    for attr in sorted(attrs, key=lambda a: a.confidence, reverse=True):
        existing_key = _find_similar_attr_key(attr.name, result)
        if existing_key is None:
            result[attr.name] = attr
    return list(result.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_similar_attr_key(attr_name: str, existing: Dict) -> Optional[str]:
    """
    Return the key in `existing` that is semantically similar to attr_name,
    or None if no match. Checks exact match, word containment, and high name similarity.
    """
    norm_new = _normalise_name(attr_name)
    for key in existing:
        norm_key = _normalise_name(key)
        if norm_key == norm_new:
            return key
        # One name is a substring of the other (e.g. "available copies" in "number of available copies")
        if norm_new in norm_key or norm_key in norm_new:
            return key
        if _name_similarity(attr_name, key) >= 0.80:
            return key
    return None


def _normalise_name(name: str) -> str:
    """Lowercase, remove non-alpha, split camelCase."""
    # Split camelCase: "LibraryUser" -> "library user"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]
