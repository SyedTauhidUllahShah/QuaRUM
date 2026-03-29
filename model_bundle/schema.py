"""
Model bundle schema: B = (M, E, C, S)

M  – UMLModel          : canonical UML elements (entities + relationships)
E  – evidence mapping  : element_name -> list of verbatim quoted spans
C  – confidence scores : element_name -> float score from the scorer
S  – segmentation meta : segment_id   -> SegmentMetadata from Phase I

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ElementType(str, Enum):
    CLASS = "Class"
    INTERFACE = "Interface"
    ENUMERATION = "Enumeration"
    ACTOR = "Actor"


class RelationshipType(str, Enum):
    IS_A = "IS_A"
    IS_PART_OF = "IS_PART_OF"
    AGGREGATES = "AGGREGATES"
    IMPLEMENTS = "IMPLEMENTS"
    ASSOCIATES = "ASSOCIATES"
    DEPENDS_ON = "DEPENDS_ON"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSpan:
    """A verbatim quoted fragment from the source requirements."""
    text: str                           # exact quote from source
    segment_id: str                     # which segment this came from
    source_document: str                # filename
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    char_start: Optional[int] = None    # character offset within source doc
    char_end: Optional[int] = None


# ---------------------------------------------------------------------------
# Model elements
# ---------------------------------------------------------------------------

@dataclass
class Attribute:
    name: str
    type: str = "String"
    owner: str = ""                     # name of the owning entity
    evidence: List[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Operation:
    name: str
    parameters: List[str] = field(default_factory=list)   # ["param: Type", ...]
    return_type: str = "void"
    owner: str = ""
    evidence: List[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Entity:
    name: str                           # in-vivo, singular noun
    definition: str
    element_type: ElementType = ElementType.CLASS
    attributes: List[Attribute] = field(default_factory=list)
    operations: List[Operation] = field(default_factory=list)
    evidence: List[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0
    aliases: List[str] = field(default_factory=list)       # merged variant names


@dataclass
class Relationship:
    relationship_type: RelationshipType
    source: str                         # source entity name
    target: str                         # target entity name
    multiplicity_source: str = "1"
    multiplicity_target: str = "1"
    role_name: Optional[str] = None     # in-vivo verb/preposition from text
    evidence: List[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Canonical UML model  (M)
# ---------------------------------------------------------------------------

@dataclass
class UMLModel:
    """M: read-only canonical UML model after selective coding."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Segmentation metadata  (S)
# ---------------------------------------------------------------------------

@dataclass
class SegmentMetadata:
    """Metadata attached to each segment by Phase I."""
    segment_id: str
    source_document: str
    format: str                         # "txt", "pdf", "docx", "md"
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    nesting_depth: int = 0
    parent_heading: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_count: int = 0


@dataclass
class Segment:
    """A processed text segment produced by Phase I."""
    segment_id: str
    text: str
    metadata: SegmentMetadata


# ---------------------------------------------------------------------------
# Model bundle  B = (M, E, C, S)
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """
    The complete output of QuaRUM and the direct input to E-QuaRUM.

    model       : M  – canonical UML model
    evidence    : E  – element_name -> list[EvidenceSpan]
    confidence  : C  – element_name -> float
    segmentation: S  – segment_id   -> SegmentMetadata
    source_corpus: path(s) to source documents used
    """
    model: UMLModel
    evidence: Dict[str, List[EvidenceSpan]]
    confidence: Dict[str, float]
    segmentation: Dict[str, SegmentMetadata]
    source_corpus: str
