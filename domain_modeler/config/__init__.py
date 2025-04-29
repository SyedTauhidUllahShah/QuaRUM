"""
Configuration components for domain modeling.

This package provides settings and configuration management
for the domain modeling framework.
"""

from domain_modeler.config.settings import Settings
from domain_modeler.config.prompts import PromptTemplates

__all__ = [
    'Settings',
    'PromptTemplates'
]