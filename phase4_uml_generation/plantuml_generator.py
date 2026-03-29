"""
Phase IV – PlantUML Generator.

Translates the validated UML model into PlantUML syntax conforming to
UML 2.5.1 specifications.

Correct PlantUML arrow notation (verified against plantuml.com/class-diagram):
  Generalization (IS_A)   : Child --|> Parent   (open triangle at parent)
  Realization (IMPLEMENTS): Class ..|> Interface (dashed, open triangle at interface)
  Composition (IS_PART_OF): Whole *-- Part       (filled diamond at whole)
  Aggregation (AGGREGATES): Aggregate o-- Part   (hollow diamond at aggregate)
  Association (ASSOCIATES): A -- B  or  A --> B  (with direction)
  Dependency (DEPENDS_ON) : Client ..> Target    (dashed arrow)

Multiplicities are always quoted: "1", "0..*", "0..10"

Uses the `plantuml` Python package for optional PNG rendering.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from model_bundle.schema import (
    Attribute,
    ElementType,
    Entity,
    Operation,
    Relationship,
    RelationshipType,
    UMLModel,
)
from .layout_optimizer import assign_packages, order_entities_for_layout


# ---------------------------------------------------------------------------
# Correct PlantUML arrow notation
# ---------------------------------------------------------------------------
# For IS_A:    source=Child, target=Parent  -> "Child --|> Parent"
# For IMPL:    source=Class, target=Iface   -> "Class ..|> Interface"
# For PART_OF: source=Whole, target=Part    -> "Whole *-- Part"
# For AGGR:    source=Aggregate, target=Part -> "Aggregate o-- Part"
# For ASSOC:   source, target               -> "src --> tgt"
# For DEP:     source depends on target     -> "src ..> tgt"

_REL_NOTATION: Dict[RelationshipType, str] = {
    RelationshipType.IS_A: "--|>",        # Child --|> Parent (open triangle at parent)
    RelationshipType.IS_PART_OF: "*--",   # Whole *-- Part   (filled diamond at whole)
    RelationshipType.AGGREGATES: "o--",   # Agg o-- Part     (hollow diamond at aggregate)
    RelationshipType.IMPLEMENTS: "..|>",  # Class ..|> Iface (dashed, open triangle at iface)
    RelationshipType.ASSOCIATES: "-->",   # A --> B          (directed association)
    RelationshipType.DEPENDS_ON: "..>",   # A ..> B          (dashed dependency)
}


class PlantUMLGenerator:

    def generate(self, model: UMLModel, title: Optional[str] = None) -> str:
        """
        Generate complete PlantUML source from the UML model.
        Returns the .puml content as a string.
        """
        lines: List[str] = ["@startuml"]

        # Global layout hints
        lines.append("skinparam classAttributeIconSize 0")
        lines.append("skinparam monochrome false")
        lines.append("skinparam shadowing false")
        lines.append("hide circle")
        lines.append("")

        if title:
            lines.append(f'title {_sanitize_title(title)}')
            lines.append("")

        # Layout pass: order entities and assign packages
        ordered_entities = order_entities_for_layout(model.entities, model.relationships)
        packages = assign_packages(ordered_entities)

        # Emit packages
        for pkg_name, entities in packages.items():
            lines.append(f'package "{pkg_name}" {{')
            for entity in entities:
                lines.extend(self._render_entity(entity, indent="  "))
                lines.append("")
            lines.append("}")
            lines.append("")

        # Emit relationships (after all class definitions)
        for rel in model.relationships:
            line = self._render_relationship(rel)
            if line:
                lines.append(line)

        lines.append("")
        lines.append("@enduml")
        return "\n".join(lines)

    def render_to_png(self, puml_content: str, output_path: str) -> bool:
        """
        Optionally render the PlantUML source to PNG using the plantuml package.
        Returns True if successful, False if the plantuml package is unavailable.
        """
        try:
            from plantuml import PlantUML
            server = PlantUML(url="http://www.plantuml.com/plantuml/img/")
            png_data = server.processes(puml_content)
            if png_data:
                with open(output_path, "wb") as f:
                    f.write(png_data)
                return True
        except ImportError:
            pass
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Entity rendering
    # ------------------------------------------------------------------

    def _render_entity(self, entity: Entity, indent: str = "") -> List[str]:
        lines: List[str] = []
        keyword = _element_keyword(entity.element_type)
        safe_name = _sanitize_identifier(entity.name)

        # Actors rendered as class with <<Actor>> stereotype (actor keyword
        # doesn't support attribute blocks in PlantUML class diagrams)
        stereotype = ""
        if entity.element_type == ElementType.INTERFACE:
            stereotype = " <<interface>>"
        elif entity.element_type == ElementType.ACTOR:
            stereotype = " <<Actor>>"

        lines.append(f"{indent}{keyword} {safe_name}{stereotype} {{")

        # Attributes section
        if entity.attributes:
            for attr in entity.attributes:
                lines.append(f"{indent}  {_render_attribute(attr)}")

        # Separator between attributes and operations
        if entity.attributes and entity.operations:
            lines.append(f"{indent}  --")

        # Operations section
        if entity.operations:
            for op in entity.operations:
                lines.append(f"{indent}  {_render_operation(op)}")

        lines.append(f"{indent}}}")
        return lines

    # ------------------------------------------------------------------
    # Relationship rendering
    # ------------------------------------------------------------------

    def _render_relationship(self, rel: Relationship) -> Optional[str]:
        notation = _REL_NOTATION.get(rel.relationship_type)
        if notation is None:
            return None

        src = _sanitize_identifier(rel.source)
        tgt = _sanitize_identifier(rel.target)

        if rel.relationship_type in (RelationshipType.IS_A, RelationshipType.IMPLEMENTS):
            # source=Child/Class, target=Parent/Interface
            # "Child --|> Parent"  or  "Class ..|> Interface"
            return f"{src} {notation} {tgt}"

        if rel.relationship_type in (RelationshipType.IS_PART_OF,
                                      RelationshipType.AGGREGATES):
            # source=Whole/Aggregate, target=Part
            # "Whole *-- Part"  or  "Aggregate o-- Part"
            # Include multiplicities when non-trivial
            mult_src = _quote_mult(rel.multiplicity_source)
            mult_tgt = _quote_mult(rel.multiplicity_target)
            label = f' : {rel.role_name}' if rel.role_name else ''
            if mult_src or mult_tgt:
                return f"{src} {mult_src}{notation}{mult_tgt} {tgt}{label}".strip()
            return f"{src} {notation} {tgt}{label}".strip()

        # ASSOCIATES and DEPENDS_ON
        mult_src = _quote_mult(rel.multiplicity_source)
        mult_tgt = _quote_mult(rel.multiplicity_target)
        label = f' : {rel.role_name}' if rel.role_name else ''

        if mult_src or mult_tgt:
            return f'{src} {mult_src} {notation} {mult_tgt} {tgt}{label}'.strip()
        return f'{src} {notation} {tgt}{label}'.strip()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _element_keyword(element_type: ElementType) -> str:
    return {
        ElementType.CLASS: "class",
        ElementType.INTERFACE: "interface",
        ElementType.ENUMERATION: "enum",
        ElementType.ACTOR: "class",  # actor keyword invalid with attribute blocks; use class + <<Actor>> stereotype
    }.get(element_type, "class")


def _render_attribute(attr: Attribute) -> str:
    safe_name = _sanitize_identifier(attr.name)
    uml_type = _map_uml_type(attr.type)
    return f"+{safe_name} : {uml_type}"


def _render_operation(op: Operation) -> str:
    safe_name = _sanitize_identifier(op.name.split("(")[0])
    params = _render_params(op.parameters)
    ret = _map_uml_type(op.return_type)
    return f"+{safe_name}({params}) : {ret}"


def _render_params(parameters: List[str]) -> str:
    """Render operation parameters, sanitising each name:type pair."""
    if not parameters:
        return ""
    rendered = []
    for p in parameters:
        if ":" in p:
            pname, ptype = p.split(":", 1)
            rendered.append(f"{_sanitize_identifier(pname.strip())} : {_map_uml_type(ptype.strip())}")
        else:
            rendered.append(_sanitize_identifier(p.strip()))
    return ", ".join(rendered)


def _map_uml_type(raw_type: str) -> str:
    """Normalise type strings to UML 2.5.1 primitive types."""
    if not raw_type:
        return "String"
    mapping = {
        "str": "String", "string": "String", "text": "String",
        "int": "Integer", "integer": "Integer", "number": "Integer",
        "float": "Float", "double": "Float", "decimal": "Float",
        "bool": "Boolean", "boolean": "Boolean",
        "date": "Date", "datetime": "DateTime", "time": "Time",
        "void": "void", "none": "void",
        "list": "List", "dict": "Map", "set": "Set",
    }
    return mapping.get(raw_type.lower().strip(), raw_type.strip())


def _quote_mult(mult: str) -> str:
    """Quote a multiplicity value for PlantUML if it is non-trivial."""
    if not mult or mult == "1":
        return ""
    return f'"{mult}" '


def _sanitize_identifier(name: str) -> str:
    """Ensure identifier is valid in PlantUML (no spaces, special chars)."""
    # Split on spaces -> camelCase
    parts = name.strip().split()
    if len(parts) > 1:
        name = parts[0] + "".join(p.capitalize() for p in parts[1:])
    # Remove remaining invalid chars
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Must not start with a digit
    if name and name[0].isdigit():
        name = "_" + name
    return name


def _sanitize_title(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 _\-]", "", title).strip()
