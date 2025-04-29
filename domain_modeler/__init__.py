"""
Domain Modeler - A research-oriented framework for AI-driven domain modeling.

This package provides tools and methods for automatically extracting
domain models from natural language requirements using qualitative 
coding techniques and large language models.
"""

__version__ = "0.1.0"
__author__ = "Domain Modeler Team"

from domain_modeler.core.enums import CSLRelationshipType
from domain_modeler.core.code import Code
from domain_modeler.core.relationship import CodeRelationship
from domain_modeler.core.code_system import CodeSystem