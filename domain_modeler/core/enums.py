"""
Enumeration types used in domain modeling.

This module defines enumerations for relationship types and other
categorical data used throughout the domain modeling process.
"""

from enum import Enum

class CSLRelationshipType(Enum):
    """
    Enumeration of relationship types between domain model elements.
    
    These relationship types represent the standard UML relationships
    and domain-specific relationships that may appear in requirements.
    """
    IS_A = "is a"
    IS_PART_OF = "is part of"
    PERFORMS = "performs"
    IMPLEMENTATION = "implements"
    DEPENDS_ON = "depends on"
    USES = "uses"
    ASSOCIATION = "associates with"
    OWNS = "owns"
    MANAGES = "manages"
    CREATES = "creates"
    ACCESSES = "accesses"