"""
Code system container module.

This module defines the CodeSystem class, which serves as the 
container for the entire domain model, managing all code elements
and their relationships.
"""

import json
from typing import Dict, List, Tuple, Optional, Set
from difflib import SequenceMatcher

from quarum.core.code import Code
from quarum.core.relationship import CodeRelationship
from quarum.core.enums import CSLRelationshipType

class CodeSystem:
    """
    Container for the entire domain model.
    
    The CodeSystem manages a collection of Code objects and their
    relationships, providing methods to add, find, and manipulate
    the model elements. It also stores domain-specific metadata
    extracted during analysis.
    """
    
    def __init__(self):
        """Initialize a new empty domain model container."""
        self.codes: Dict[str, Code] = {}
        self.relationships: List[CodeRelationship] = []
        self.domain_description = ""
        self.core_domain_entities: List[str] = []
        self.domain_actors: List[str] = []
        self.domain_attributes: Dict[str, List[str]] = {}
        self.domain_operations: Dict[str, List[str]] = {}
        self.enumerations: Dict[str, List[str]] = {}
        self.domain_relationships = {}
        self.raw_text = ""
    
    def add_code(self, code: Code) -> str:
        """
        Add a code element to the system, handling duplicates intelligently.
        
        This method checks for existing similar code elements to avoid
        duplicating entities with minor naming differences, and merges
        attributes and other properties when appropriate.
        
        Args:
            code: The Code object to add to the system
            
        Returns:
            The ID of the added or merged code element
        """
        # Check for existing similar code elements
        for eid, existing_code in self.codes.items():
            # Check for exact name match or high similarity
            if (existing_code.name.lower() == code.name.lower() or
                SequenceMatcher(None, existing_code.name.lower(), code.name.lower()).ratio() > 0.9):
                
                # Merge attributes (ensuring uniqueness)
                existing_code.attributes = list(set(
                    json.dumps(attr, sort_keys=True) for attr in existing_code.attributes + code.attributes
                ))
                existing_code.attributes = [json.loads(attr) for attr in existing_code.attributes]
                
                # Merge methods (ensuring uniqueness)
                existing_code.methods = list(set(
                    json.dumps(meth, sort_keys=True) for meth in existing_code.methods + code.methods
                ))
                existing_code.methods = [json.loads(meth) for meth in existing_code.methods]
                
                # Merge stereotypes
                existing_code.stereotypes = list(set(existing_code.stereotypes + code.stereotypes))
                
                # Extend evidence
                existing_code.evidence_chunks.extend(code.evidence_chunks)
                existing_code.evidence_locations.extend(code.evidence_locations)
                
                # Use higher confidence
                existing_code.confidence = max(existing_code.confidence, code.confidence)
                
                # Update flags
                if code.is_interface:
                    existing_code.is_interface = True
                if code.is_abstract:
                    existing_code.is_abstract = True
                if code.is_enumeration:
                    existing_code.is_enumeration = True
                    existing_code.enum_values = list(set(existing_code.enum_values + code.enum_values))
                    
                # Update definition if needed
                if code.definition and not existing_code.definition:
                    existing_code.definition = code.definition
                    
                # Update extracted text if available
                if code.extracted_text:
                    existing_code.extracted_text = code.extracted_text
                
                # Return the existing ID
                return eid
        
        # If no match found, add as new code
        self.codes[code.id] = code
        return code.id
    
    def add_relationship(self, relationship: CodeRelationship) -> bool:
        """
        Add a relationship to the system, validating and handling duplicates.
        
        This method checks if both ends of the relationship exist in the 
        system, validates relationship types for semantic correctness, and
        handles duplicate relationships by merging or updating as appropriate.
        
        Args:
            relationship: The CodeRelationship to add
            
        Returns:
            True if added as a new relationship, False if updated existing or invalid
        """
        # Validate that both codes exist
        if relationship.source_code_id not in self.codes or relationship.target_code_id not in self.codes:
            return False
        
        source = self.codes[relationship.source_code_id]
        target = self.codes[relationship.target_code_id]
        
        # Validate implementation relationships
        if relationship.relationship_type == CSLRelationshipType.IMPLEMENTATION and not target.is_interface:
            return False
        
        # Check for existing relationships
        for r in self.relationships:
            if (r.source_code_id == relationship.source_code_id and 
                r.target_code_id == relationship.target_code_id and 
                r.relationship_type == relationship.relationship_type):
                
                # Update existing relationship if confidence is higher
                if relationship.confidence > r.confidence:
                    r.confidence = relationship.confidence
                    r.association_name = relationship.association_name
                    r.multiplicity = relationship.multiplicity
                    r.evidence_chunks.extend(relationship.evidence_chunks)
                    r.evidence_locations.extend(relationship.evidence_locations)
                    r.extracted_text = relationship.extracted_text
                    
                return False  # Relationship already exists
        
        # Add new relationship
        self.relationships.append(relationship)
        source.outgoing_relationships.append(relationship)
        target.incoming_relationships.append(relationship)
        return True
    
    def calculate_relevance_scores(self) -> None:
        """
        Calculate relevance scores for all code elements.
        
        This assigns higher scores to elements that:
        - Have more relationships (connected to more elements)
        - Are core domain entities
        - Have higher confidence
        - Are abstract or interfaces (likely more important architecturally)
        """
        for code_id, code in self.codes.items():
            # Base score on number of relationships
            relation_count = len(code.incoming_relationships) + len(code.outgoing_relationships)
            code.relevance_score = min(relation_count * 0.2, 2.0)
            
            # Bonus for core domain entities
            if code.name.lower() in [c.lower() for c in self.core_domain_entities]:
                code.relevance_score += 1.0
            
            # Factor in confidence
            code.relevance_score += code.confidence * 0.5
            
            # Bonus for interfaces and abstract classes
            if code.is_interface or code.is_abstract:
                code.relevance_score += 0.3
    
    def get_codes_by_stereotype(self, stereotype: str) -> List[Code]:
        """
        Get code elements with a specific stereotype.
        
        Args:
            stereotype: The stereotype to filter by (e.g., "entity", "actor")
            
        Returns:
            List of Code objects with the specified stereotype
        """
        return [code for code in self.codes.values() 
                if stereotype in code.stereotypes]
    
    def get_code_by_name(self, name: str) -> Optional[Code]:
        """
        Find a code element by name (case-insensitive).
        
        Args:
            name: The name to search for
            
        Returns:
            The matching Code object or None if not found
        """
        name_lower = name.lower()
        for code in self.codes.values():
            if code.name.lower() == name_lower:
                return code
        return None
    
    def get_relationships_by_type(self, rel_type: CSLRelationshipType) -> List[CodeRelationship]:
        """
        Get relationships of a specific type.
        
        Args:
            rel_type: The relationship type to filter by
            
        Returns:
            List of CodeRelationship objects of the specified type
        """
        return [rel for rel in self.relationships 
                if rel.relationship_type == rel_type]
    
    def get_inheritance_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get the inheritance hierarchy in the model.
        
        Returns:
            Dictionary mapping parent code IDs to lists of child code IDs
        """
        hierarchy = {}
        for rel in self.relationships:
            if rel.relationship_type == CSLRelationshipType.IS_A:
                parent_id = rel.target_code_id
                child_id = rel.source_code_id
                
                if parent_id not in hierarchy:
                    hierarchy[parent_id] = []
                    
                hierarchy[parent_id].append(child_id)
                
        return hierarchy