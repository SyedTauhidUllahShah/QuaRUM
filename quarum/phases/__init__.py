"""
Qualitative analysis phases for domain modeling.

This package implements the three main phases of qualitative coding
used in domain model extraction:

1. Open Coding: Identifying and extracting domain entities
2. Axial Coding: Discovering relationships between entities
3. Selective Coding: Refining the model and establishing hierarchies

These phases are based on Grounded Theory methodology adapted for
domain modeling with natural language processing techniques.
"""

from quarum.phases.phase_utils import PhaseContext, PhaseResult
from quarum.phases.open_coding import OpenCodingPhase
from quarum.phases.axial_coding import AxialCodingPhase
from quarum.phases.selective_coding import SelectiveCodingPhase

__all__ = [
    'PhaseContext',
    'PhaseResult',
    'OpenCodingPhase',
    'AxialCodingPhase', 
    'SelectiveCodingPhase'
]