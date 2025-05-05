"""
Relationship representation module.

This module defines the CodeRelationship class, which represents
relationships between code elements in the domain model.
"""

from typing import Optional
from quarum.core.enums import CSLRelationshipType


class CodeRelationship:
    """
    Represents a relationship between two code elements.

    Relationships can be inheritance, implementation, association,
    composition, and other UML relationship types. Each relationship
    has a source and target code element, and includes metadata about
    how it was identified in the requirements.
    """

    def __init__(
        self,
        relationship_id: str,
        source_code_id: str,
        target_code_id: str,
        relationship_type: CSLRelationshipType,
        association_name: str = "",
        confidence: float = 0.7,
        multiplicity: Optional[dict[str, str]] = None,
    ):
        """
        Initialize a new relationship between code elements.

        Args:
            relation_id: Unique identifier for this relationship
            source_code_id: ID of the source code element
            target_code_id: ID of the target code element
            relationship_type: Type of relationship (from CSLRelationshipType)
            association_name: Name label for the association (if applicable)
            confidence: Confidence score (0.0-1.0) for this relationship
            multiplicity: Multiplicity constraints at each end of the relationship
        """
        self.id = relationship_id
        self.source_code_id = source_code_id
        self.target_code_id = target_code_id
        self.relationship_type = relationship_type
        self.association_name = association_name
        self.confidence = confidence
        self.multiplicity = (
            multiplicity if multiplicity else {"source": "1", "target": "*"}
        )
        self.evidence_chunks: list[str] = []
        self.evidence_locations: list[str] = []
        self.extracted_text: str = ""

    def add_evidence(self, chunk: str, location: Optional[str] = None) -> None:
        """
        Add evidence of this relationship from the requirements.

        Args:
            chunk: Text excerpt supporting this relationship
            location: Source location reference (e.g., file:line)
        """
        self.evidence_chunks.append(chunk)
        if location:
            self.evidence_locations.append(location)

    def set_multiplicity(self, source_mult: str, target_mult: str) -> None:
        """
        Set multiplicity constraints for this relationship.

        Args:
            source_mult: Multiplicity at the source end (e.g., "1", "0..*")
            target_mult: Multiplicity at the target end
        """
        self.multiplicity = {"source": source_mult, "target": target_mult}

    def is_inheritance(self) -> bool:
        """Check if this relationship represents inheritance."""
        return self.relationship_type == CSLRelationshipType.IS_A

    def is_implementation(self) -> bool:
        """Check if this relationship represents interface implementation."""
        return self.relationship_type == CSLRelationshipType.IMPLEMENTATION

    def is_composition(self) -> bool:
        """Check if this relationship represents composition."""
        return self.relationship_type == CSLRelationshipType.IS_PART_OF
