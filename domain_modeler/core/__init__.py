"""
Core domain modeling components.

This package contains the foundational data structures and classes
for representing domain models, including code elements and their
relationships.
"""

from domain_modeler.core.enums import CSLRelationshipType
from domain_modeler.core.code import Code
from domain_modeler.core.relationship import CodeRelationship
from domain_modeler.core.code_system import CodeSystem

__all__ = [
    'CSLRelationshipType',
    'Code',
    'CodeRelationship',
    'CodeSystem'
]