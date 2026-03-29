"""
Serialises and deserialises ModelBundle B=(M,E,C,S) to/from JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .schema import (
    Attribute,
    ElementType,
    Entity,
    EvidenceSpan,
    ModelBundle,
    Operation,
    Relationship,
    RelationshipType,
    Segment,
    SegmentMetadata,
    UMLModel,
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _evidence_to_dict(e: EvidenceSpan) -> Dict[str, Any]:
    return {
        "text": e.text,
        "segment_id": e.segment_id,
        "source_document": e.source_document,
        "section_title": e.section_title,
        "page_number": e.page_number,
        "char_start": e.char_start,
        "char_end": e.char_end,
    }


def _attribute_to_dict(a: Attribute) -> Dict[str, Any]:
    return {
        "name": a.name,
        "type": a.type,
        "owner": a.owner,
        "evidence": [_evidence_to_dict(e) for e in a.evidence],
        "confidence": a.confidence,
    }


def _operation_to_dict(op: Operation) -> Dict[str, Any]:
    return {
        "name": op.name,
        "parameters": op.parameters,
        "return_type": op.return_type,
        "owner": op.owner,
        "evidence": [_evidence_to_dict(e) for e in op.evidence],
        "confidence": op.confidence,
    }


def _entity_to_dict(ent: Entity) -> Dict[str, Any]:
    return {
        "name": ent.name,
        "definition": ent.definition,
        "element_type": ent.element_type.value,
        "attributes": [_attribute_to_dict(a) for a in ent.attributes],
        "operations": [_operation_to_dict(op) for op in ent.operations],
        "evidence": [_evidence_to_dict(e) for e in ent.evidence],
        "confidence": ent.confidence,
        "aliases": ent.aliases,
    }


def _relationship_to_dict(rel: Relationship) -> Dict[str, Any]:
    return {
        "relationship_type": rel.relationship_type.value,
        "source": rel.source,
        "target": rel.target,
        "multiplicity_source": rel.multiplicity_source,
        "multiplicity_target": rel.multiplicity_target,
        "role_name": rel.role_name,
        "evidence": [_evidence_to_dict(e) for e in rel.evidence],
        "confidence": rel.confidence,
    }


def _metadata_to_dict(m: SegmentMetadata) -> Dict[str, Any]:
    return {
        "segment_id": m.segment_id,
        "source_document": m.source_document,
        "format": m.format,
        "section_title": m.section_title,
        "page_number": m.page_number,
        "nesting_depth": m.nesting_depth,
        "parent_heading": m.parent_heading,
        "char_start": m.char_start,
        "char_end": m.char_end,
        "token_count": m.token_count,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_bundle(bundle: ModelBundle, output_path: str) -> None:
    """Write the ModelBundle to a JSON file at output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "source_corpus": bundle.source_corpus,
        "model": {
            "entities": [_entity_to_dict(e) for e in bundle.model.entities],
            "relationships": [_relationship_to_dict(r) for r in bundle.model.relationships],
        },
        "evidence": {
            k: [_evidence_to_dict(ev) for ev in v]
            for k, v in bundle.evidence.items()
        },
        "confidence": bundle.confidence,
        "segmentation": {
            k: _metadata_to_dict(v)
            for k, v in bundle.segmentation.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Deserialisation
# ---------------------------------------------------------------------------

def _dict_to_evidence(d: Dict[str, Any]) -> EvidenceSpan:
    return EvidenceSpan(
        text=d["text"],
        segment_id=d["segment_id"],
        source_document=d["source_document"],
        section_title=d.get("section_title"),
        page_number=d.get("page_number"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
    )


def _dict_to_attribute(d: Dict[str, Any]) -> Attribute:
    return Attribute(
        name=d["name"],
        type=d.get("type", "String"),
        owner=d.get("owner", ""),
        evidence=[_dict_to_evidence(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
    )


def _dict_to_operation(d: Dict[str, Any]) -> Operation:
    return Operation(
        name=d["name"],
        parameters=d.get("parameters", []),
        return_type=d.get("return_type", "void"),
        owner=d.get("owner", ""),
        evidence=[_dict_to_evidence(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
    )


def _dict_to_entity(d: Dict[str, Any]) -> Entity:
    return Entity(
        name=d["name"],
        definition=d.get("definition", ""),
        element_type=ElementType(d.get("element_type", "Class")),
        attributes=[_dict_to_attribute(a) for a in d.get("attributes", [])],
        operations=[_dict_to_operation(op) for op in d.get("operations", [])],
        evidence=[_dict_to_evidence(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
        aliases=d.get("aliases", []),
    )


def _dict_to_relationship(d: Dict[str, Any]) -> Relationship:
    return Relationship(
        relationship_type=RelationshipType(d.get("relationship_type", "ASSOCIATES")),
        source=d["source"],
        target=d["target"],
        multiplicity_source=d.get("multiplicity_source", "1"),
        multiplicity_target=d.get("multiplicity_target", "1"),
        role_name=d.get("role_name"),
        evidence=[_dict_to_evidence(e) for e in d.get("evidence", [])],
        confidence=d.get("confidence", 0.0),
    )


def _dict_to_metadata(d: Dict[str, Any]) -> SegmentMetadata:
    return SegmentMetadata(
        segment_id=d["segment_id"],
        source_document=d["source_document"],
        format=d.get("format", "txt"),
        section_title=d.get("section_title"),
        page_number=d.get("page_number"),
        nesting_depth=d.get("nesting_depth", 0),
        parent_heading=d.get("parent_heading"),
        char_start=d.get("char_start"),
        char_end=d.get("char_end"),
        token_count=d.get("token_count", 0),
    )


def load_bundle(path: str) -> ModelBundle:
    """Load a ModelBundle from a JSON file produced by export_bundle."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model = UMLModel(
        entities=[_dict_to_entity(e) for e in payload["model"]["entities"]],
        relationships=[_dict_to_relationship(r) for r in payload["model"]["relationships"]],
    )

    evidence = {
        k: [_dict_to_evidence(ev) for ev in v]
        for k, v in payload.get("evidence", {}).items()
    }

    confidence = payload.get("confidence", {})

    segmentation = {
        k: _dict_to_metadata(v)
        for k, v in payload.get("segmentation", {}).items()
    }

    return ModelBundle(
        model=model,
        evidence=evidence,
        confidence=confidence,
        segmentation=segmentation,
        source_corpus=payload.get("source_corpus", ""),
    )
