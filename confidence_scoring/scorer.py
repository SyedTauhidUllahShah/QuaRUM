"""
Confidence Scoring Mechanism (Section 3.2.3 of the paper).

Scoring is performed SEPARATELY from generation so that the component
proposing items does not evaluate its own outputs.

Four signals:
  E – Exact text match:        required terms appear verbatim in quoted evidence
  F – Facet coverage:          fraction of required structure that has direct evidence
  R – Retrieval support:       mean cosine similarity of supporting segments (normalised)
  C – Cross-segment consistency: fraction of top-k segments that agree with the claim

Phase-specific profiles:
  score_open   = clip(0.50E + 0.30F + 0.15R + 0.05C)
  score_axial  = clip(0.35E + 0.35F + 0.15R + 0.05C + 0.10S)
  score_sel    = clip(0.70*score_prior + 0.30*U)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import config
from model_bundle.schema import (
    Attribute,
    ElementType,
    Entity,
    EvidenceSpan,
    Operation,
    Relationship,
    RelationshipType,
)


# ---------------------------------------------------------------------------
# Public scorer class
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Stateless scorer.  All methods are pure functions over their inputs.
    No access to the LLM or retrieval components.
    """

    # ------------------------------------------------------------------
    # Open coding: entities
    # ------------------------------------------------------------------

    def score_entity(
        self,
        entity: Entity,
        evidence_spans: List[EvidenceSpan],
        retrieved_texts: List[Tuple[str, float]],   # [(segment_text, cosine_score)]
    ) -> float:
        """
        score_open = clip(0.50E + 0.30F + 0.15R + 0.05C)
        Facets for entities: name + at least one attribute or operation.
        """
        E = self._exact_match_entity(entity, evidence_spans)
        F = self._facet_coverage_entity(entity, evidence_spans)
        R = self._retrieval_support(retrieved_texts)
        C = self._cross_segment_consistency_entity(entity, retrieved_texts)

        raw = (
            config.SCORE_OPEN_E * E
            + config.SCORE_OPEN_F * F
            + config.SCORE_OPEN_R * R
            + config.SCORE_OPEN_C * C
        )
        return _clip(raw)

    def score_attribute(
        self,
        attr: Attribute,
        owner_name: str,
        evidence_spans: List[EvidenceSpan],
        retrieved_texts: List[Tuple[str, float]],
    ) -> float:
        """
        Facets for attributes: owner present in evidence AND attr name in quoted text.
        """
        E = self._term_in_evidence(attr.name, evidence_spans)
        F = self._facet_coverage_attribute(attr, owner_name, evidence_spans)
        R = self._retrieval_support(retrieved_texts)
        C = self._term_in_retrieved(attr.name, retrieved_texts)

        raw = (
            config.SCORE_OPEN_E * E
            + config.SCORE_OPEN_F * F
            + config.SCORE_OPEN_R * R
            + config.SCORE_OPEN_C * C
        )
        return _clip(raw)

    def score_operation(
        self,
        op: Operation,
        owner_name: str,
        evidence_spans: List[EvidenceSpan],
        retrieved_texts: List[Tuple[str, float]],
    ) -> float:
        """
        Facets for operations: owner present AND operation verb in quoted text.
        """
        E = self._term_in_evidence(op.name.split("(")[0], evidence_spans)
        F = self._facet_coverage_operation(op, owner_name, evidence_spans)
        R = self._retrieval_support(retrieved_texts)
        C = self._term_in_retrieved(op.name.split("(")[0], retrieved_texts)

        raw = (
            config.SCORE_OPEN_E * E
            + config.SCORE_OPEN_F * F
            + config.SCORE_OPEN_R * R
            + config.SCORE_OPEN_C * C
        )
        return _clip(raw)

    # ------------------------------------------------------------------
    # Axial coding: relationships
    # ------------------------------------------------------------------

    def score_relationship(
        self,
        rel: Relationship,
        evidence_spans: List[EvidenceSpan],
        retrieved_texts: List[Tuple[str, float]],
    ) -> float:
        """
        score_axial = clip(0.35E + 0.35F + 0.15R + 0.05C + 0.10S)

        Facets: both endpoints + relationship cue + multiplicity when stated.
        If endpoints or cue are absent from evidence, score_axial = 0.
        """
        E = self._exact_match_relationship(rel, evidence_spans)
        F = self._facet_coverage_relationship(rel, evidence_spans)

        # Hard rule: if core facets missing, score = 0
        if F == 0.0:
            return 0.0

        R = self._retrieval_support(retrieved_texts)
        C = self._cross_segment_consistency_relationship(rel, retrieved_texts)
        S = self._structural_check_relationship(rel)

        raw = (
            config.SCORE_AXIAL_E * E
            + config.SCORE_AXIAL_F * F
            + config.SCORE_AXIAL_R * R
            + config.SCORE_AXIAL_C * C
            + config.SCORE_AXIAL_S * S
        )
        return _clip(raw)

    # ------------------------------------------------------------------
    # Selective coding: model validation
    # ------------------------------------------------------------------

    def score_selective(
        self,
        prior_score: float,
        uml_checks_passed: int,
        uml_checks_total: int,
        structural_violation: bool = False,
    ) -> float:
        """
        score_sel = clip(0.70*score_prior + 0.30*U)
        U = fraction of UML checks passed.
        Any structural violation sets U = 0 and blocks promotion.
        """
        if structural_violation:
            U = 0.0
        else:
            U = uml_checks_passed / uml_checks_total if uml_checks_total > 0 else 1.0

        raw = config.SCORE_SEL_PRIOR * prior_score + config.SCORE_SEL_UML * U
        return _clip(raw)

    # ------------------------------------------------------------------
    # Signal: E – Exact text match
    # ------------------------------------------------------------------

    def _exact_match_entity(
        self, entity: Entity, evidence_spans: List[EvidenceSpan]
    ) -> float:
        """1.0 if entity name appears verbatim in any evidence span."""
        return self._term_in_evidence(entity.name, evidence_spans)

    def _exact_match_relationship(
        self, rel: Relationship, evidence_spans: List[EvidenceSpan]
    ) -> float:
        """Both entity names present in combined evidence text."""
        combined = " ".join(e.text for e in evidence_spans).lower()
        source_present = rel.source.lower() in combined
        target_present = rel.target.lower() in combined
        if source_present and target_present:
            return 1.0
        if source_present or target_present:
            return 0.5
        return 0.0

    def _term_in_evidence(
        self, term: str, evidence_spans: List[EvidenceSpan]
    ) -> float:
        if not term or not evidence_spans:
            return 0.0
        combined = " ".join(e.text for e in evidence_spans).lower()
        # Match whole-word or camelCase prefix
        pattern = re.compile(r"\b" + re.escape(term.lower()), re.IGNORECASE)
        return 1.0 if pattern.search(combined) else 0.0

    # ------------------------------------------------------------------
    # Signal: F – Facet coverage
    # ------------------------------------------------------------------

    def _facet_coverage_entity(
        self, entity: Entity, evidence_spans: List[EvidenceSpan]
    ) -> float:
        """
        Facets: entity name + at least one attribute OR one operation.
        Returns fraction of facets with direct evidence.
        """
        facets_supported = 0
        total_facets = 0

        # Facet 1: entity name
        total_facets += 1
        if self._term_in_evidence(entity.name, evidence_spans) > 0:
            facets_supported += 1

        # Facet 2+: each attribute name
        for attr in entity.attributes:
            total_facets += 1
            if self._term_in_evidence(attr.name, evidence_spans) > 0:
                facets_supported += 1

        # Facet 3+: each operation verb
        for op in entity.operations:
            total_facets += 1
            verb = op.name.split("(")[0]
            if self._term_in_evidence(verb, evidence_spans) > 0:
                facets_supported += 1

        if total_facets == 0:
            return 0.0

        # Must have entity name + at least one structural facet
        name_ok = self._term_in_evidence(entity.name, evidence_spans) > 0
        has_structure = (len(entity.attributes) > 0 or len(entity.operations) > 0)
        if not name_ok:
            return 0.0
        if has_structure and facets_supported < 2:
            return facets_supported / total_facets * 0.5   # penalise

        return facets_supported / total_facets

    def _facet_coverage_attribute(
        self,
        attr: Attribute,
        owner_name: str,
        evidence_spans: List[EvidenceSpan],
    ) -> float:
        """Facets: owner name present AND attribute name in quoted text."""
        owner_ok = self._term_in_evidence(owner_name, evidence_spans)
        attr_ok = self._term_in_evidence(attr.name, evidence_spans)
        if owner_ok and attr_ok:
            return 1.0
        if owner_ok or attr_ok:
            return 0.5
        return 0.0

    def _facet_coverage_operation(
        self,
        op: Operation,
        owner_name: str,
        evidence_spans: List[EvidenceSpan],
    ) -> float:
        """Facets: owner present AND operation verb in quoted text."""
        owner_ok = self._term_in_evidence(owner_name, evidence_spans)
        verb = op.name.split("(")[0]
        verb_ok = self._term_in_evidence(verb, evidence_spans)
        if owner_ok and verb_ok:
            return 1.0
        if owner_ok or verb_ok:
            return 0.5
        return 0.0

    def _facet_coverage_relationship(
        self, rel: Relationship, evidence_spans: List[EvidenceSpan]
    ) -> float:
        """
        Facets: both endpoints + a cue word.
        Endpoints and cue MUST be present; otherwise returns 0.
        """
        combined = " ".join(e.text for e in evidence_spans).lower()

        source_ok = rel.source.lower() in combined
        target_ok = rel.target.lower() in combined
        cue_ok = rel.role_name and rel.role_name.lower() in combined

        if not (source_ok and target_ok):
            return 0.0   # hard requirement

        facets_total = 3  # endpoints (2) + cue (1)
        facets_ok = 2     # both endpoints confirmed

        if cue_ok:
            facets_ok += 1

        # Multiplicity facet (optional)
        mult_target = rel.multiplicity_target or "1"
        if mult_target not in ("1", ""):
            facets_total += 1
            mult_terms = [t for t in ["up to", "at least", "one or more", "zero or more",
                          "many", mult_target] if t is not None]
            if any(t in combined for t in mult_terms):
                facets_ok += 1

        return facets_ok / facets_total

    # ------------------------------------------------------------------
    # Signal: R – Retrieval support
    # ------------------------------------------------------------------

    def _retrieval_support(
        self, retrieved_texts: List[Tuple[str, float]]
    ) -> float:
        """Mean cosine similarity of supporting segments, normalised to [0,1]."""
        if not retrieved_texts:
            return 0.0
        scores = [score for _, score in retrieved_texts]
        return min(1.0, sum(scores) / len(scores))

    # ------------------------------------------------------------------
    # Signal: C – Cross-segment consistency
    # ------------------------------------------------------------------

    def _cross_segment_consistency_entity(
        self, entity: Entity, retrieved_texts: List[Tuple[str, float]]
    ) -> float:
        """Fraction of top-k segments that mention the entity name."""
        if not retrieved_texts:
            return 0.0
        name_lower = entity.name.lower()
        agreeing = sum(1 for text, _ in retrieved_texts if name_lower in text.lower())
        return agreeing / len(retrieved_texts)

    def _cross_segment_consistency_relationship(
        self, rel: Relationship, retrieved_texts: List[Tuple[str, float]]
    ) -> float:
        """Fraction of top-k segments that mention both entity names."""
        if not retrieved_texts:
            return 0.0
        s, t = rel.source.lower(), rel.target.lower()
        agreeing = sum(
            1 for text, _ in retrieved_texts
            if s in text.lower() and t in text.lower()
        )
        return agreeing / len(retrieved_texts)

    def _term_in_retrieved(
        self, term: str, retrieved_texts: List[Tuple[str, float]]
    ) -> float:
        if not retrieved_texts or not term:
            return 0.0
        term_lower = term.lower()
        found = sum(1 for text, _ in retrieved_texts if term_lower in text.lower())
        return found / len(retrieved_texts)

    # ------------------------------------------------------------------
    # Signal: S – Structural check (axial only)
    # ------------------------------------------------------------------

    def _structural_check_relationship(self, rel: Relationship) -> float:
        """
        S in [0,1]: directionality and type compatibility.
        Checks:
          - IS_A: always directional (child -> parent)
          - IMPLEMENTS: class -> interface
          - IS_PART_OF: composition structure
          - Others: role_name present indicates directionality was asserted
        """
        rt = rel.relationship_type
        if rt == RelationshipType.NONE:
            return 0.0
        if rt in (RelationshipType.IS_A, RelationshipType.IMPLEMENTS):
            return 1.0   # inherently directional
        if rt in (RelationshipType.IS_PART_OF, RelationshipType.AGGREGATES):
            return 0.9   # composition/aggregation implies containment direction
        if rel.role_name:
            return 0.8   # cue present
        return 0.5       # direction asserted without explicit cue


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
