"""
Checkpoint save/load for each pipeline phase.
Allows resuming a pipeline run from any phase.

Checkpoint files are stored under output/checkpoints/<doc_stem>/:
  phase1_segments.json       – Phase I output (List[Segment])
  phase2_vectorstore/        – Phase II output (VectorStore, saved via its own save/load)
  phase3a_entities.json      – Phase III open coding output (List[Entity])
  phase3b_relationships.json – Phase III axial coding output (List[Relationship])
  phase3c_entities.json      – Phase III selective coding final entities (List[Entity])
  phase3c_relationships.json – Phase III selective coding final relationships (List[Relationship])
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import List, Tuple

from model_bundle.schema import (
    Attribute,
    ElementType,
    Entity,
    EvidenceSpan,
    Operation,
    Relationship,
    RelationshipType,
    Segment,
    SegmentMetadata,
)

logger = logging.getLogger(__name__)

_CHECKPOINTS_ROOT = Path("output") / "checkpoints"

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def checkpoint_dir(doc_stem: str) -> Path:
    """Return the checkpoint directory for a given document stem."""
    return _CHECKPOINTS_ROOT / doc_stem


def phase_exists(doc_stem: str, phase: int) -> bool:
    """
    Check whether the checkpoint for *phase* already exists on disk.

    Phase mapping:
      1 – phase1_segments.json
      2 – phase2_vectorstore/index.faiss
      3 – phase3c_entities.json + phase3c_relationships.json
      4 – (UML generation has no checkpoint; always reruns)
    """
    d = checkpoint_dir(doc_stem)
    if phase == 1:
        return (d / "phase1_segments.json").exists()
    if phase == 2:
        return (d / "phase2_vectorstore" / "index.faiss").exists()
    if phase == 3:
        return (
            (d / "phase3c_entities.json").exists()
            and (d / "phase3c_relationships.json").exists()
        )
    return False


# ---------------------------------------------------------------------------
# Low-level JSON helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Serialisation helpers (dataclass → plain dict)
# ---------------------------------------------------------------------------

def _evidence_span_to_dict(span: EvidenceSpan) -> dict:
    return dataclasses.asdict(span)


def _evidence_span_from_dict(d: dict) -> EvidenceSpan:
    return EvidenceSpan(
        text=d["text"],
        segment_id=d["segment_id"],
        source_document=d["source_document"],
        section_title=d.get("section_title"),
        page_number=d.get("page_number"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
    )


def _attribute_to_dict(attr: Attribute) -> dict:
    return {
        "name": attr.name,
        "type": attr.type,
        "owner": attr.owner,
        "evidence": [_evidence_span_to_dict(e) for e in attr.evidence],
        "confidence": attr.confidence,
    }


def _attribute_from_dict(d: dict) -> Attribute:
    return Attribute(
        name=d["name"],
        type=d.get("type", "String"),
        owner=d.get("owner", ""),
        evidence=[_evidence_span_from_dict(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
    )


def _operation_to_dict(op: Operation) -> dict:
    return {
        "name": op.name,
        "parameters": op.parameters,
        "return_type": op.return_type,
        "owner": op.owner,
        "evidence": [_evidence_span_to_dict(e) for e in op.evidence],
        "confidence": op.confidence,
    }


def _operation_from_dict(d: dict) -> Operation:
    return Operation(
        name=d["name"],
        parameters=d.get("parameters", []),
        return_type=d.get("return_type", "void"),
        owner=d.get("owner", ""),
        evidence=[_evidence_span_from_dict(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
    )


def _entity_to_dict(entity: Entity) -> dict:
    return {
        "name": entity.name,
        "definition": entity.definition,
        "element_type": entity.element_type.value,
        "attributes": [_attribute_to_dict(a) for a in entity.attributes],
        "operations": [_operation_to_dict(o) for o in entity.operations],
        "evidence": [_evidence_span_to_dict(e) for e in entity.evidence],
        "confidence": entity.confidence,
        "aliases": entity.aliases,
    }


def _entity_from_dict(d: dict) -> Entity:
    return Entity(
        name=d["name"],
        definition=d.get("definition", ""),
        element_type=ElementType(d.get("element_type", ElementType.CLASS.value)),
        attributes=[_attribute_from_dict(a) for a in d.get("attributes", [])],
        operations=[_operation_from_dict(o) for o in d.get("operations", [])],
        evidence=[_evidence_span_from_dict(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
        aliases=d.get("aliases", []),
    )


def _relationship_to_dict(rel: Relationship) -> dict:
    return {
        "relationship_type": rel.relationship_type.value,
        "source": rel.source,
        "target": rel.target,
        "multiplicity_source": rel.multiplicity_source,
        "multiplicity_target": rel.multiplicity_target,
        "role_name": rel.role_name,
        "evidence": [_evidence_span_to_dict(e) for e in rel.evidence],
        "confidence": rel.confidence,
    }


def _relationship_from_dict(d: dict) -> Relationship:
    return Relationship(
        relationship_type=RelationshipType(d["relationship_type"]),
        source=d["source"],
        target=d["target"],
        multiplicity_source=d.get("multiplicity_source", "1"),
        multiplicity_target=d.get("multiplicity_target", "1"),
        role_name=d.get("role_name"),
        evidence=[_evidence_span_from_dict(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
    )


def _segment_metadata_to_dict(meta: SegmentMetadata) -> dict:
    return dataclasses.asdict(meta)


def _segment_metadata_from_dict(d: dict) -> SegmentMetadata:
    return SegmentMetadata(
        segment_id=d["segment_id"],
        source_document=d["source_document"],
        format=d["format"],
        section_title=d.get("section_title"),
        page_number=d.get("page_number"),
        nesting_depth=d.get("nesting_depth", 0),
        parent_heading=d.get("parent_heading"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
        token_count=d.get("token_count", 0),
    )


def _segment_to_dict(seg: Segment) -> dict:
    return {
        "segment_id": seg.segment_id,
        "text": seg.text,
        "metadata": _segment_metadata_to_dict(seg.metadata),
    }


def _segment_from_dict(d: dict) -> Segment:
    return Segment(
        segment_id=d["segment_id"],
        text=d["text"],
        metadata=_segment_metadata_from_dict(d["metadata"]),
    )


# ---------------------------------------------------------------------------
# Public save/load API
# ---------------------------------------------------------------------------

# ---- Phase I ---------------------------------------------------------------

def save_phase1(doc_stem: str, segments: List[Segment]) -> None:
    """Serialize Phase I segments to JSON."""
    path = checkpoint_dir(doc_stem) / "phase1_segments.json"
    data = [_segment_to_dict(s) for s in segments]
    _write_json(path, data)
    logger.debug("Checkpoint: saved %d segments to %s", len(segments), path)


def load_phase1(doc_stem: str) -> List[Segment]:
    """Load Phase I segments from checkpoint."""
    path = checkpoint_dir(doc_stem) / "phase1_segments.json"
    data = _read_json(path)
    segments = [_segment_from_dict(d) for d in data]
    logger.debug("Checkpoint: loaded %d segments from %s", len(segments), path)
    return segments


# ---- Phase III-a (open coding) ---------------------------------------------

def save_phase3a(doc_stem: str, entities: List[Entity]) -> None:
    """Serialize open-coding entities to JSON."""
    path = checkpoint_dir(doc_stem) / "phase3a_entities.json"
    data = [_entity_to_dict(e) for e in entities]
    _write_json(path, data)
    logger.debug("Checkpoint: saved %d entities (3a) to %s", len(entities), path)


def load_phase3a(doc_stem: str) -> List[Entity]:
    """Load open-coding entities from checkpoint."""
    path = checkpoint_dir(doc_stem) / "phase3a_entities.json"
    data = _read_json(path)
    entities = [_entity_from_dict(d) for d in data]
    logger.debug("Checkpoint: loaded %d entities (3a) from %s", len(entities), path)
    return entities


# ---- Phase III-b (axial coding) --------------------------------------------

def save_phase3b(doc_stem: str, relationships: List[Relationship]) -> None:
    """Serialize axial-coding relationships to JSON."""
    path = checkpoint_dir(doc_stem) / "phase3b_relationships.json"
    data = [_relationship_to_dict(r) for r in relationships]
    _write_json(path, data)
    logger.debug(
        "Checkpoint: saved %d relationships (3b) to %s", len(relationships), path
    )


def load_phase3b(doc_stem: str) -> List[Relationship]:
    """Load axial-coding relationships from checkpoint."""
    path = checkpoint_dir(doc_stem) / "phase3b_relationships.json"
    data = _read_json(path)
    relationships = [_relationship_from_dict(d) for d in data]
    logger.debug(
        "Checkpoint: loaded %d relationships (3b) from %s", len(relationships), path
    )
    return relationships


# ---- Phase III-c (selective coding – final) --------------------------------

def save_phase3c(
    doc_stem: str, entities: List[Entity], relationships: List[Relationship]
) -> None:
    """Serialize final selective-coding entities and relationships to JSON."""
    d = checkpoint_dir(doc_stem)
    _write_json(d / "phase3c_entities.json", [_entity_to_dict(e) for e in entities])
    _write_json(
        d / "phase3c_relationships.json",
        [_relationship_to_dict(r) for r in relationships],
    )
    logger.debug(
        "Checkpoint: saved %d entities, %d relationships (3c) to %s",
        len(entities), len(relationships), d,
    )


def load_phase3c(
    doc_stem: str,
) -> Tuple[List[Entity], List[Relationship]]:
    """Load final selective-coding entities and relationships from checkpoint."""
    d = checkpoint_dir(doc_stem)
    entities = [_entity_from_dict(x) for x in _read_json(d / "phase3c_entities.json")]
    relationships = [
        _relationship_from_dict(x)
        for x in _read_json(d / "phase3c_relationships.json")
    ]
    logger.debug(
        "Checkpoint: loaded %d entities, %d relationships (3c) from %s",
        len(entities), len(relationships), d,
    )
    return entities, relationships
