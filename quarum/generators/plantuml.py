"""
PlantUML diagram generation module.

This module provides the PlantUMLGenerator class for converting
a CodeSystem into PlantUML diagram code for visualization.
"""

import re
import time
from typing import Dict, List, Any, Optional

from quarum.core.code_system import CodeSystem
from quarum.core.code import Code
from quarum.core.enums import CSLRelationshipType


class PlantUMLGenerator:
    """
    Generator for PlantUML diagrams from domain models.
    
    This class converts a CodeSystem object into PlantUML code that
    can be rendered into a UML class diagram.
    """
    
    def __init__(self, code_system: CodeSystem, domain_name: str):
        """
        Initialize the PlantUML generator.
        
        Args:
            code_system: The domain model code system
            domain_name: Name of the domain
        """
        self.code_system = code_system
        self.domain_name = domain_name
        
    def generate(self, style: Optional[str] = None) -> str:
        """
        Generate PlantUML code for the domain model.
        
        Args:
            style: Optional styling preset ('default', 'monochrome', 'vibrant')
            
        Returns:
            PlantUML code as string
        """
        # Use appropriate styling
        style_lines = self._get_style_directives(style or 'default')
        
        # Start with PlantUML header and style
        lines = ["@startuml", f"' {self.domain_name} Domain Model"]
        lines.extend(style_lines)
        lines.append("")
        
        # Categorize entities
        actors = []
        interfaces = []
        abstract_classes = []
        enumerations = []
        concrete_classes = []
        
        for code in self.code_system.codes.values():
            if self._is_banned_term(code.name):
                continue
                
            if "actor" in code.stereotypes:
                actors.append(code)
            elif code.is_enumeration:
                enumerations.append(code)
            elif code.is_interface:
                interfaces.append(code)
            elif code.is_abstract:
                abstract_classes.append(code)
            else:
                concrete_classes.append(code)
        
        # Generate packages for each category
        self._add_entity_package(lines, "Actors", actors, "actor")
        self._add_entity_package(lines, "Interfaces", interfaces, "interface")
        self._add_entity_package(lines, "AbstractClasses", abstract_classes, "abstract class")
        self._add_entity_package(lines, "Enumerations", enumerations, "enum")
        self._add_entity_package(lines, "Entities", concrete_classes, "class")
        
        # Add relationships
        lines.append("")
        self._add_relationships(lines)
        
        # Add notes and footer
        lines.append("")
        lines.append("note as ModelSource")
        lines.append(f"  Domain model for {self.domain_name}")
        lines.append("  Generated from requirements using AI-driven analysis")
        lines.append(f"  Created on {time.strftime('%Y-%m-%d')}")
        lines.append("end note")
        
        # End diagram
        lines.append("@enduml")
        
        return "\n".join(lines)
    
    def _get_style_directives(self, style: str) -> List[str]:
        """
        Get PlantUML style directives for the diagram.
        
        Args:
            style: Style name ('default', 'monochrome', 'vibrant')
            
        Returns:
            List of style directive lines
        """
        styles = {
            'default': [
                "skinparam monochrome false",
                "skinparam classAttributeIconSize 0",
                "skinparam packageStyle rectangle",
                "skinparam shadowing false",
                "skinparam class {",
                "  BackgroundColor white",
                "  ArrowColor black",
                "  BorderColor black",
                "  FontSize 12",
                "}",
            ],
            'monochrome': [
                "skinparam monochrome true",
                "skinparam classAttributeIconSize 0", 
                "skinparam packageStyle rectangle",
                "skinparam shadowing false",
                "skinparam class {",
                "  BackgroundColor white",
                "  ArrowColor black",
                "  BorderColor black",
                "  FontSize 12",
                "}",
            ],
            'vibrant': [
                "skinparam monochrome false",
                "skinparam classAttributeIconSize 0",
                "skinparam packageStyle rectangle",
                "skinparam shadowing true",
                "skinparam class {",
                "  BackgroundColor lightyellow",
                "  ArrowColor #33a6b8",
                "  BorderColor #33a6b8",
                "  FontSize 12",
                "}",
                "skinparam interface {",
                "  BackgroundColor lightblue",
                "  BorderColor #2c9ad1",
                "}",
                "skinparam enum {",
                "  BackgroundColor lightgreen",
                "  BorderColor green",
                "}",
                "skinparam abstractClass {",
                "  BackgroundColor #f8d6d6",
                "  BorderColor #e63946",
                "}",
            ],
        }
        
        return styles.get(style, styles['default'])
    
    def _add_entity_package(
        self, 
        lines: List[str], 
        package_name: str, 
        entities: List[Code],
        entity_type: str
    ) -> None:
        """
        Add a package with entities to the diagram.
        
        Args:
            lines: List of diagram lines (modified in place)
            package_name: Name of the package
            entities: List of entities to add
            entity_type: Type of entity ('class', 'interface', etc.)
        """
        if not entities:
            return
            
        lines.append(f"package {package_name} {{")
        
        # Add stereotype map for this package
        stereotype_map = {
            "interface": "<<interface>>",
            "abstract": "<<abstract>>",
            "class": "<<class>>",
            "enumeration": "<<enumeration>>",
            "actor": "<<actor>>"
        }
        
        # Add each entity
        for entity in sorted(entities, key=lambda x: x.name):
            # Get stereotypes
            stereotypes = [stereotype_map[s] for s in entity.stereotypes if s in stereotype_map]
            stereotype_str = " ".join(stereotypes) if stereotypes else ""
            
            # Add entity declaration
            sanitized_name = self._sanitize_name(entity.name)
            lines.append(f"  {entity_type} {sanitized_name} {stereotype_str} {{")
            
            # Add attributes for non-enums
            if not entity.is_enumeration:
                for attr in entity.attributes:
                    lines.append(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
            
            # Add methods for non-enums
            if not entity.is_enumeration:
                for method in entity.methods:
                    lines.append(f"    {method['visibility']} {method['name']}{method['signature']}")
            
            # Add enum values
            if entity.is_enumeration:
                for value in entity.enum_values:
                    lines.append(f"    {value}")
            
            # Close entity declaration
            lines.append("  }")
        
        lines.append("}")
    
    def _add_relationships(self, lines: List[str]) -> None:
        """
        Add relationships to the diagram.
        
        Args:
            lines: List of diagram lines (modified in place)
        """
        for rel in sorted(self.code_system.relationships, key=lambda r: r.id):
            source = self.code_system.codes.get(rel.source_code_id)
            target = self.code_system.codes.get(rel.target_code_id)
            
            # Skip if either entity doesn't exist
            if not source or not target:
                continue
                
            # Skip if either entity name is banned
            if self._is_banned_term(source.name) or self._is_banned_term(target.name):
                continue
                
            source_name = self._sanitize_name(source.name)
            target_name = self._sanitize_name(target.name)
            
            # Generate relationship based on type
            if rel.relationship_type == CSLRelationshipType.IS_A:
                lines.append(f"{source_name} --|> {target_name}")
                
            elif rel.relationship_type == CSLRelationshipType.IMPLEMENTATION:
                lines.append(f"{source_name} ..|> {target_name}")
                
            elif rel.relationship_type == CSLRelationshipType.IS_PART_OF:
                lines.append(
                    f"{source_name} *-- {target_name} : {rel.association_name or 'contains'}"
                )
                
            elif rel.relationship_type in [CSLRelationshipType.DEPENDS_ON, CSLRelationshipType.USES]:
                lines.append(
                    f"{source_name} ..> {target_name} : {rel.association_name or 'uses'}"
                )
                
            elif rel.relationship_type == CSLRelationshipType.MANAGES:
                source_mult = rel.multiplicity.get("source", "1")
                target_mult = rel.multiplicity.get("target", "*")
                lines.append(
                    f'{source_name} "{source_mult}" --> "{target_mult}" {target_name} : manages'
                )
                
            elif rel.relationship_type == CSLRelationshipType.CREATES:
                source_mult = rel.multiplicity.get("source", "1")
                target_mult = rel.multiplicity.get("target", "*")
                lines.append(
                    f'{source_name} "{source_mult}" --> "{target_mult}" {target_name} : creates'
                )
                
            else:
                source_mult = rel.multiplicity.get("source", "1")
                target_mult = rel.multiplicity.get("target", "*")
                lines.append(
                    f'{source_name} "{source_mult}" -- "{target_mult}" {target_name} : {rel.association_name or ""}'
                )
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a name for use in PlantUML.
        
        Args:
            name: Original name
            
        Returns:
            Sanitized name
        """
        # Replace spaces with underscores
        result = re.sub(r"\s+", "_", name)
        
        # Remove non-alphanumeric characters
        result = re.sub(r"[^a-zA-Z0-9_]", "", result)
        
        # Ensure name doesn't start with a digit
        if result and result[0].isdigit():
            result = "_" + result
            
        # Provide a fallback for empty names
        return result if result else "Item"
    
    def _is_banned_term(self, name: str) -> bool:
        """
        Check if a name is a banned generic term.
        
        Args:
            name: Name to check
            
        Returns:
            True if name is banned, False otherwise
        """
        banned_words = set(["system", "component", "generic", "entity", "item", "object", "module"])
        nl = name.strip().lower()
        
        # Check if in banned words set
        if nl in banned_words:
            return True
        
        # Check if too short
        if len(nl) <= 2:
            return True
            
        return False