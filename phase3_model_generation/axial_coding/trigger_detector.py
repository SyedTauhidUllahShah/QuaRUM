"""
Axial Coding – Trigger Detector.

Four triggers that work together to identify potential relationships
between validated entities in a segment:

  1. Co-mention: two entity names appear in the same sentence/paragraph
  2. In-vivo cues: explicit relationship words between entity names
     ("borrows", "contains", "is a type of", "depends on", etc.)
  3. Attribute/parameter type: an attribute or operation names the other entity
  4. Container phrases: "catalog of", "list of", "set of" + plural noun

Returns a list of trigger hits: (entity1, entity2, cue_word, trigger_type)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from model_bundle.schema import Entity


# In-vivo cue lexicon covering the IoT domain
_CUE_WORDS = [
    # Inheritance / specialisation
    "is a", "is an", "is a type of", "type of", "kind of", "subtype",
    "extends", "inherits", "specializes",
    # Composition / part-of
    "consists of", "is composed of", "is part of", "contains",
    "includes", "has a", "has an", "made up of",
    # Aggregation
    "aggregates", "comprises", "groups", "collects", "manages",
    # Association / dependency verbs  (IoT-specific additions)
    "borrows", "uses", "controls", "monitors", "tracks", "registers",
    "assigns", "sends", "receives", "triggers", "activates", "deactivates",
    "reports", "belongs to", "associated with", "connected to", "paired with",
    "installed in", "assigned to", "enrolled in", "linked to",
    # Interface
    "implements", "supports",
    # General dependency
    "depends on", "requires", "needs",
]

# Container phrase patterns: "catalog of X", "list of X", "set of X"
_CONTAINER_PATTERN = re.compile(
    r"\b(?:catalog|list|set|collection|group|registry|log|record|history|"
    r"directory|library|database|store|pool|queue|index)\s+of\s+(\w+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TriggerHit = Tuple[str, str, str, str]  # (entity1_name, entity2_name, cue, trigger_type)


def detect_triggers(
    entities: List[Entity],
    segment_text: str,
    neighbor_texts: List[str],
) -> List[TriggerHit]:
    """
    Scan segment and neighbours for the four trigger types.
    Returns deduplicated list of (entity1, entity2, cue, trigger_type).
    """
    all_text = segment_text + "\n" + "\n".join(neighbor_texts)
    entity_names = [e.name for e in entities]

    hits: List[TriggerHit] = []
    seen: Set[Tuple[str, str]] = set()

    # Build name->entity index
    entity_map: Dict[str, Entity] = {e.name: e for e in entities}

    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i >= j:
                continue

            pair_key = (e1.name, e2.name)
            if pair_key in seen:
                continue

            # Trigger 1: co-mention in same sentence/paragraph
            cue, ttype = _check_co_mention(e1.name, e2.name, all_text)
            if cue:
                hits.append((e1.name, e2.name, cue, ttype))
                seen.add(pair_key)
                continue

            # Trigger 3: attribute/parameter type
            cue = _check_attribute_type(e1, e2)
            if cue:
                hits.append((e1.name, e2.name, cue, "attribute_type"))
                seen.add(pair_key)
                continue

            # Trigger 4: container phrase
            cue = _check_container_phrase(e1.name, e2.name, all_text)
            if cue:
                hits.append((e1.name, e2.name, cue, "container"))
                seen.add(pair_key)

    return hits


# ---------------------------------------------------------------------------
# Individual trigger implementations
# ---------------------------------------------------------------------------

def _check_co_mention(
    name1: str, name2: str, text: str
) -> Tuple[Optional[str], str]:
    """
    Trigger 1 + Trigger 2 combined.
    Returns (cue_word, trigger_type) or (None, "").
    Splits text into sentences; checks if both names co-occur.
    If yes, looks for an in-vivo cue between them.
    """
    # Split into sentences (rough split on ". ")
    sentences = re.split(r"(?<=[.!?])\s+", text)

    n1, n2 = name1.lower(), name2.lower()

    for sentence in sentences:
        s_lower = sentence.lower()
        if n1 not in s_lower or n2 not in s_lower:
            continue

        # Trigger 1: co-mention confirmed
        # Trigger 2: look for cue word between the two names
        for cue in _CUE_WORDS:
            if cue in s_lower:
                return cue, "cue_word"

        # Co-mention without explicit cue
        return "co-mention", "co_mention"

    return None, ""


def _check_attribute_type(entity1: Entity, entity2: Entity) -> Optional[str]:
    """
    Trigger 3: entity1 has an attribute or operation whose type or
    parameter references entity2's name.
    """
    n2 = entity2.name.lower()
    for attr in entity1.attributes:
        if n2 in attr.type.lower() or n2 in attr.name.lower():
            return f"attribute:{attr.name}"
    for op in entity1.operations:
        params_str = " ".join(op.parameters).lower()
        if n2 in params_str or n2 in op.return_type.lower():
            return f"operation:{op.name}"
    return None


def _check_container_phrase(
    name1: str, name2: str, text: str
) -> Optional[str]:
    """
    Trigger 4: container phrase like "catalog of books" binds name1 and name2.
    """
    for match in _CONTAINER_PATTERN.finditer(text):
        contained = match.group(1).lower()
        # Check if the contained noun matches either entity
        if contained.startswith(name2.lower()[:4]) or name2.lower() in contained:
            # Verify name1 appears near the container phrase
            ctx_start = max(0, match.start() - 100)
            ctx_end = min(len(text), match.end() + 100)
            context = text[ctx_start:ctx_end].lower()
            if name1.lower() in context:
                return match.group(0)
    return None
