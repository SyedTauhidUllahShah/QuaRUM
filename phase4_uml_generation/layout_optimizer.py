"""
Phase IV – Layout Optimizer.

Groups model elements into packages for clarity.
Determines rendering order to:
  - Place inheritance hierarchies vertically
  - Keep composition/aggregation pairs near each other
  - Separate actors, interfaces, enumerations from concrete classes

Returns an ordered list of entities per package group.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from model_bundle.schema import ElementType, Entity, Relationship, RelationshipType


# Package names (from paper: actors, interfaces, abstract classes, enumerations, concrete entities)
PKG_ACTORS = "Actors"
PKG_INTERFACES = "Interfaces"
PKG_ENUMERATIONS = "Enumerations"
PKG_ENTITIES = "Entities"


def assign_packages(entities: List[Entity]) -> Dict[str, List[Entity]]:
    """
    Assign each entity to a package based on its element type.
    Returns dict: package_name -> list of entities.
    """
    packages: Dict[str, List[Entity]] = {
        PKG_ACTORS: [],
        PKG_INTERFACES: [],
        PKG_ENUMERATIONS: [],
        PKG_ENTITIES: [],
    }
    for entity in entities:
        if entity.element_type == ElementType.ACTOR:
            packages[PKG_ACTORS].append(entity)
        elif entity.element_type == ElementType.INTERFACE:
            packages[PKG_INTERFACES].append(entity)
        elif entity.element_type == ElementType.ENUMERATION:
            packages[PKG_ENUMERATIONS].append(entity)
        else:
            packages[PKG_ENTITIES].append(entity)

    # Remove empty packages
    return {k: v for k, v in packages.items() if v}


def order_entities_for_layout(
    entities: List[Entity],
    relationships: List[Relationship],
) -> List[Entity]:
    """
    Order entities so that:
      1. Parents appear before children (IS_A hierarchy)
      2. Containers appear before contained (IS_PART_OF / AGGREGATES)
      3. Remaining entities follow original order
    """
    parent_map: Dict[str, str] = {}
    for rel in relationships:
        if rel.relationship_type in (RelationshipType.IS_A,
                                      RelationshipType.IS_PART_OF,
                                      RelationshipType.AGGREGATES):
            parent_map[rel.source] = rel.target

    ordered: List[Entity] = []
    visited: Set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        # Visit parent first
        if name in parent_map:
            visit(parent_map[name])
        # Add this entity
        entity = next((e for e in entities if e.name == name), None)
        if entity:
            ordered.append(entity)

    for entity in entities:
        visit(entity.name)

    return ordered
