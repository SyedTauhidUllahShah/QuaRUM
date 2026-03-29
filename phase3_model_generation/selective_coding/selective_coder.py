"""
Selective Coding – Algorithm 3 from the paper.

Ticketed, item-scoped refinement. Does NOT re-run open/axial globally.

Loop until convergence:
  1. validate_structure(M)     -> structural tickets
  2. For each structural ticket: targeted retrieval + LLM fix + score_sel
  3. detect_low_coverage(M)    -> coverage tickets
  4. For each coverage ticket:  targeted retrieval + LLM enrich + score_sel
  5. prune_low_confidence(M, τ)
  6. merge_duplicates(M)
  7. Check convergence criteria

Convergence:
  - pending tickets == 0
  - avg_confidence_change < 0.05
  (pipeline additionally checks entity rate and relationship stability)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

import config
from confidence_scoring.scorer import ConfidenceScorer
from model_bundle.schema import (
    Attribute,
    Entity,
    EvidenceSpan,
    Operation,
    Relationship,
    RelationshipType,
    UMLModel,
)
from phase2_knowledge_construction.retriever import Retriever
from phase3_model_generation.open_coding.merger import merge_duplicate_entities
from .ticket_manager import (
    IssueType,
    Ticket,
    TicketCategory,
    TicketManager,
    TicketStatus,
)
from .prompts import (
    SELECTIVE_ENRICH_SYSTEM,
    SELECTIVE_ENRICH_USER_TEMPLATE,
    SELECTIVE_FIX_SYSTEM,
    SELECTIVE_FIX_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class SelectiveCoder:
    def __init__(self, retriever: Retriever, scorer: ConfidenceScorer):
        self._retriever = retriever
        self._scorer = scorer
        self._client = OpenAI()
        self._ticket_manager = TicketManager()

    # ------------------------------------------------------------------
    # Algorithm 3: Selective Coding
    # ------------------------------------------------------------------

    def run(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Execute selective coding until convergence.
        Returns (refined_entities, refined_relationships).
        """
        model = UMLModel(entities=list(entities), relationships=list(relationships))
        prev_avg_confidence = _avg_confidence(model)

        for iteration in range(config.SELECTIVE_MAX_ITERATIONS):
            logger.info("Selective coding iteration %d", iteration + 1)

            # --- Structural validation ---
            struct_tickets = self._validate_structure(model)
            for ticket in struct_tickets:
                self._resolve_ticket(ticket, model)

            # --- Coverage enrichment ---
            coverage_tickets = self._detect_low_coverage(model)
            for ticket in coverage_tickets:
                self._resolve_ticket(ticket, model)

            # --- Prune low confidence ---
            model = _prune_low_confidence(model, config.CONFIDENCE_THRESHOLD)

            # --- Merge duplicates ---
            model.entities = merge_duplicate_entities(model.entities)

            # --- Check convergence ---
            curr_avg = _avg_confidence(model)
            delta = abs(curr_avg - prev_avg_confidence)
            pending = self._ticket_manager.pending_count()

            logger.info(
                "  Iteration %d: pending=%d, avg_conf=%.4f, delta=%.4f",
                iteration + 1, pending, curr_avg, delta
            )

            if pending == 0 and delta < config.CONVERGENCE_CONFIDENCE_DELTA:
                logger.info("Selective coding converged at iteration %d", iteration + 1)
                break

            prev_avg_confidence = curr_avg
            self._ticket_manager.clear_resolved()

        return model.entities, model.relationships

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------

    def _validate_structure(self, model: UMLModel) -> List[Ticket]:
        tickets: List[Ticket] = []
        entity_names = {e.name for e in model.entities}

        # Check 1: circular inheritance
        for chain in _detect_inheritance_cycles(model):
            t = self._ticket_manager.create(
                category=TicketCategory.STRUCTURAL,
                issue_type=IssueType.CIRCULAR_INHERITANCE,
                affected_elements=chain,
                description=f"Circular IS_A chain: {' -> '.join(chain)}",
            )
            tickets.append(t)

        # Check 2: invalid multiplicities
        for rel in model.relationships:
            for mult, label in [
                (rel.multiplicity_source, "source"),
                (rel.multiplicity_target, "target"),
            ]:
                if not _valid_multiplicity(mult):
                    t = self._ticket_manager.create(
                        category=TicketCategory.STRUCTURAL,
                        issue_type=IssueType.INVALID_MULTIPLICITY,
                        affected_elements=[rel.source, rel.target],
                        description=f"Invalid {label} multiplicity '{mult}' on {rel.source}->{rel.target}",
                    )
                    tickets.append(t)

        # Check 3: dangling references (relationship endpoints not in entity set)
        for rel in model.relationships:
            missing = []
            if rel.source not in entity_names:
                missing.append(rel.source)
            if rel.target not in entity_names:
                missing.append(rel.target)
            if missing:
                t = self._ticket_manager.create(
                    category=TicketCategory.STRUCTURAL,
                    issue_type=IssueType.DANGLING_REFERENCE,
                    affected_elements=missing,
                    description=f"Relationship references unknown entities: {missing}",
                )
                tickets.append(t)

        return tickets

    # ------------------------------------------------------------------
    # Coverage detection
    # ------------------------------------------------------------------

    def _detect_low_coverage(self, model: UMLModel) -> List[Ticket]:
        """
        Flag entities with < 1 attribute AND < 1 operation as low coverage.
        These are candidates for enrichment via targeted retrieval.
        """
        tickets: List[Ticket] = []
        for entity in model.entities:
            if len(entity.attributes) == 0 and len(entity.operations) == 0:
                t = self._ticket_manager.create(
                    category=TicketCategory.COVERAGE,
                    issue_type=IssueType.LOW_COVERAGE,
                    affected_elements=[entity.name],
                    description=f"Entity '{entity.name}' has no attributes or operations",
                )
                tickets.append(t)
        return tickets

    # ------------------------------------------------------------------
    # Ticket resolution
    # ------------------------------------------------------------------

    def _resolve_ticket(self, ticket: Ticket, model: UMLModel) -> None:
        """Attempt to resolve a ticket using targeted retrieval + LLM + scoring."""
        # Targeted retrieval: scope to affected elements only
        query = " ".join(ticket.affected_elements)
        context_segs = self._retriever.retrieve(query, k=config.RETRIEVAL_TOP_K)
        context_text = "\n---\n".join(s.text for s, _ in context_segs[:3])
        retrieved_texts = [(s.text, score) for s, score in context_segs]

        if ticket.category == TicketCategory.STRUCTURAL:
            self._resolve_structural_ticket(ticket, model, context_text, retrieved_texts)
        else:
            self._resolve_coverage_ticket(ticket, model, context_text, retrieved_texts)

    def _resolve_structural_ticket(
        self,
        ticket: Ticket,
        model: UMLModel,
        context_text: str,
        retrieved_texts: List[Tuple[str, float]],
    ) -> None:
        element = _find_element_state(ticket.affected_elements, model)
        user_msg = SELECTIVE_FIX_USER_TEMPLATE.format(
            issue_type=ticket.issue_type.value,
            affected_elements=", ".join(ticket.affected_elements),
            issue_description=ticket.issue_description,
            context_text=context_text,
            element_state=element,
        )
        fix = self._call_llm_fix(user_msg)
        if fix is None or fix.get("action") == "none":
            self._ticket_manager.mark_unresolved(ticket)
            return

        # Score the fix
        prior = _prior_score_for_ticket(ticket.affected_elements, model)
        uml_checks = _count_uml_checks(fix)
        score = self._scorer.score_selective(
            prior_score=prior,
            uml_checks_passed=uml_checks,
            uml_checks_total=max(1, uml_checks),
            structural_violation=(ticket.issue_type == IssueType.CIRCULAR_INHERITANCE),
        )

        if score >= config.CONFIDENCE_THRESHOLD and _validate_fix_uml(fix, model):
            _apply_fix(fix, model)
            self._ticket_manager.resolve(ticket, str(fix))
            logger.debug("Resolved structural ticket %s", ticket.ticket_id)
        else:
            self._ticket_manager.mark_unresolved(ticket)

    def _resolve_coverage_ticket(
        self,
        ticket: Ticket,
        model: UMLModel,
        context_text: str,
        retrieved_texts: List[Tuple[str, float]],
    ) -> None:
        entity_name = ticket.affected_elements[0]
        entity = _find_entity(model, entity_name)
        if entity is None:
            self._ticket_manager.mark_unresolved(ticket)
            return

        user_msg = SELECTIVE_ENRICH_USER_TEMPLATE.format(
            entity_name=entity_name,
            current_attributes=", ".join(a.name for a in entity.attributes) or "none",
            current_operations=", ".join(op.name for op in entity.operations) or "none",
            context_text=context_text,
        )
        enrichment = self._call_llm_enrich(user_msg)
        if enrichment is None:
            self._ticket_manager.mark_unresolved(ticket)
            return

        new_attrs = enrichment.get("new_attributes", [])
        new_ops = enrichment.get("new_operations", [])

        added = 0
        for a_raw in new_attrs:
            attr_name = a_raw.get("name", "").strip()
            if not attr_name:
                continue
            quote = a_raw.get("evidence_quote", "")
            # Build minimal evidence for scoring
            spans = [EvidenceSpan(
                text=quote, segment_id="sel", source_document="selective"
            )] if quote else []
            attr = Attribute(name=attr_name, type=a_raw.get("type", "String"),
                             owner=entity_name, evidence=spans)
            score = self._scorer.score_attribute(
                attr, entity_name, spans, retrieved_texts
            )
            if score >= config.CONFIDENCE_THRESHOLD:
                attr.confidence = score
                entity.attributes.append(attr)
                added += 1

        for op_raw in new_ops:
            op_name = op_raw.get("name", "").strip()
            if not op_name:
                continue
            quote = op_raw.get("evidence_quote", "")
            spans = [EvidenceSpan(
                text=quote, segment_id="sel", source_document="selective"
            )] if quote else []
            op = Operation(
                name=op_name,
                parameters=op_raw.get("parameters", []),
                return_type=op_raw.get("return_type", "void"),
                owner=entity_name,
                evidence=spans,
            )
            score = self._scorer.score_operation(
                op, entity_name, spans, retrieved_texts
            )
            if score >= config.CONFIDENCE_THRESHOLD:
                op.confidence = score
                entity.operations.append(op)
                added += 1

        if added > 0:
            # Update entity confidence via selective scoring
            entity.confidence = self._scorer.score_selective(
                prior_score=entity.confidence,
                uml_checks_passed=added,
                uml_checks_total=added,
            )
            self._ticket_manager.resolve(ticket, f"Added {added} items to {entity_name}")
        else:
            self._ticket_manager.mark_unresolved(ticket)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm_fix(self, user_msg: str) -> Optional[Dict]:
        try:
            response = self._client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SELECTIVE_FIX_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return _parse_json_obj(raw)
        except Exception as exc:
            logger.warning("LLM fix call failed: %s", exc)
            return None

    def _call_llm_enrich(self, user_msg: str) -> Optional[Dict]:
        try:
            response = self._client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SELECTIVE_ENRICH_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return _parse_json_obj(raw)
        except Exception as exc:
            logger.warning("LLM enrich call failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# UML structure checks
# ---------------------------------------------------------------------------

def _detect_inheritance_cycles(model: UMLModel) -> List[List[str]]:
    """Return list of inheritance chains that form cycles."""
    parent_map: Dict[str, str] = {}
    for rel in model.relationships:
        if rel.relationship_type == RelationshipType.IS_A:
            parent_map[rel.source] = rel.target

    cycles = []
    for start in parent_map:
        visited: Set[str] = set()
        node = start
        chain = [node]
        while node in parent_map:
            node = parent_map[node]
            if node in visited:
                cycles.append(chain + [node])
                break
            visited.add(node)
            chain.append(node)
    return cycles


def _valid_multiplicity(mult: str) -> bool:
    if mult in ("1", "0..1", "*", "0..*", "1..*", ""):
        return True
    if re.match(r"^\d+\.\.\d+$", mult):
        return True
    if re.match(r"^\d+$", mult):
        return True
    return False


# ---------------------------------------------------------------------------
# Confidence and model utilities
# ---------------------------------------------------------------------------

def _avg_confidence(model: UMLModel) -> float:
    scores = [e.confidence for e in model.entities] + [r.confidence for r in model.relationships]
    return sum(scores) / len(scores) if scores else 0.0


def _prune_low_confidence(model: UMLModel, threshold: float) -> UMLModel:
    """Remove elements that remain below the threshold after selective coding."""
    entity_names_kept = {e.name for e in model.entities if e.confidence >= threshold}
    model.entities = [e for e in model.entities if e.confidence >= threshold]
    model.relationships = [
        r for r in model.relationships
        if r.confidence >= threshold
        and r.source in entity_names_kept
        and r.target in entity_names_kept
    ]
    return model


def _find_entity(model: UMLModel, name: str) -> Optional[Entity]:
    for e in model.entities:
        if e.name == name:
            return e
    return None


def _find_element_state(element_names: List[str], model: UMLModel) -> str:
    """Serialize current state of affected elements for the fix prompt."""
    parts = []
    for name in element_names:
        e = _find_entity(model, name)
        if e:
            attrs = [a.name for a in e.attributes]
            ops = [op.name for op in e.operations]
            parts.append(f"Entity {name}: attrs={attrs}, ops={ops}")
    return "\n".join(parts) if parts else "No matching elements found"


def _prior_score_for_ticket(element_names: List[str], model: UMLModel) -> float:
    scores = []
    for name in element_names:
        e = _find_entity(model, name)
        if e:
            scores.append(e.confidence)
    return sum(scores) / len(scores) if scores else 0.5


def _count_uml_checks(fix: Dict) -> int:
    """Approximate number of UML checks satisfied by a fix proposal."""
    action = fix.get("action", "none")
    if action == "none":
        return 0
    if action == "remove":
        return 1   # removing a dangling element passes 1 check
    changes = fix.get("changes", {})
    return max(1, len(changes))


def _validate_fix_uml(fix: Dict, model: UMLModel) -> bool:
    """Basic UML conformance check on a proposed fix."""
    action = fix.get("action", "none")
    if action == "none":
        return False
    if action == "remove":
        return True   # removal is always structurally valid
    changes = fix.get("changes", {})
    if "multiplicity" in changes:
        return _valid_multiplicity(str(changes["multiplicity"]))
    return True


def _apply_fix(fix: Dict, model: UMLModel) -> None:
    """Apply a validated fix to the model in-place."""
    action = fix.get("action", "none")
    element_name = fix.get("element_name", "")
    element_type = fix.get("element_type", "")
    changes = fix.get("changes", {})

    if action == "remove" and element_type == "relationship":
        model.relationships = [
            r for r in model.relationships
            if not (r.source == element_name or r.target == element_name)
        ]
    elif action == "update" and element_type == "entity":
        entity = _find_entity(model, element_name)
        if entity:
            if "name" in changes:
                entity.name = changes["name"]
    elif action == "update" and element_type == "relationship":
        for rel in model.relationships:
            key = f"{rel.source}->{rel.target}"
            if key == element_name or rel.source == element_name:
                if "multiplicity_target" in changes:
                    rel.multiplicity_target = changes["multiplicity_target"]
                if "multiplicity_source" in changes:
                    rel.multiplicity_source = changes["multiplicity_source"]
                if "relationship_type" in changes:
                    try:
                        rel.relationship_type = RelationshipType(changes["relationship_type"])
                    except ValueError:
                        pass


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json_obj(content: str) -> Optional[Dict]:
    content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
