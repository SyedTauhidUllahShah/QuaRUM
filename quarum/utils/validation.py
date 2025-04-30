"""
Model validation module.

This module provides utilities for validating and sanitizing domain models,
ensuring consistency and correctness before diagram generation.
"""

import re
import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from difflib import SequenceMatcher

from quarum.core.code_system import CodeSystem
from quarum.core.code import Code
from quarum.core.relationship import CodeRelationship
from quarum.core.enums import CSLRelationshipType

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validates and sanitizes domain models.
    
    This class checks domain models for consistency, correctness,
    and completeness, providing warnings and automatic fixes for
    common issues.
    """
    
    def __init__(self, code_system: CodeSystem):
        """
        Initialize the model validator.
        
        Args:
            code_system: The code system to validate
        """
        self.code_system = code_system
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
        
    def validate(self, auto_fix: bool = True) -> bool:
        """
        Validate the model and optionally apply fixes.
        
        Args:
            auto_fix: Whether to automatically fix issues
            
        Returns:
            True if model is valid (after fixes), False otherwise
        """
        # Reset tracking lists
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
        
        # Run validation checks
        self._check_entity_names()
        self._check_duplicate_entities()
        self._check_orphaned_entities()
        self._check_relationship_validity()
        self._check_inheritance_cycles()
        self._check_interface_implementations()
        self._check_relationship_multiplicities()
        
        # Apply fixes if requested
        if auto_fix and self.issues:
            self._apply_fixes()
            
            # Re-validate after fixes
            self.issues = []
            self.warnings = []
            
            self._check_entity_names()
            self._check_duplicate_entities()
            self._check_orphaned_entities()
            self._check_relationship_validity()
            self._check_inheritance_cycles()
            self._check_interface_implementations()
            self._check_relationship_multiplicities()
        
        # Log issues and warnings
        for issue in self.issues:
            logger.error(f"Validation issue: {issue}")
            
        for warning in self.warnings:
            logger.warning(f"Validation warning: {warning}")
            
        for fix in self.fixes_applied:
            logger.info(f"Fix applied: {fix}")
            
        # Model is valid if no issues remain
        return len(self.issues) == 0
    
    def _check_entity_names(self) -> None:
        """Check for invalid entity names."""
        for code_id, code in self.code_system.codes.items():
            # Check for empty names
            if not code.name.strip():
                self.issues.append(f"Entity with ID {code_id} has an empty name")
                
            # Check for very short names
            elif len(code.name) <= 2 and not code.name.isupper():
                self.warnings.append(f"Entity '{code.name}' has a very short name")
                
            # Check for generic names
            elif code.name.lower() in ["system", "component", "entity", "object", "item", "data"]:
                self.warnings.append(f"Entity '{code.name}' has a generic name")
                
            # Check for non-alphanumeric characters
            if not re.match(r'^[a-zA-Z0-9_\s]+$', code.name):
                self.issues.append(f"Entity '{code.name}' contains non-alphanumeric characters")
    
    def _check_duplicate_entities(self) -> None:
        """Check for duplicate entity names."""
        name_map = {}
        for code_id, code in self.code_system.codes.items():
            name_lower = code.name.lower()
            
            if name_lower in name_map:
                existing_id = name_map[name_lower]
                self.issues.append(
                    f"Duplicate entity name: '{code.name}' (IDs: {existing_id}, {code_id})"
                )
            else:
                name_map[name_lower] = code_id
                
        # Check for highly similar names
        names = list(name_map.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                similarity = SequenceMatcher(None, name1, name2).ratio()
                if similarity > 0.9:
                    self.warnings.append(
                        f"Very similar entity names: '{name1}' and '{name2}'"
                    )
    
    def _check_orphaned_entities(self) -> None:
        """Check for orphaned entities with no relationships."""
        connected_entities = set()
        
        # Collect all entities connected via relationships
        for rel in self.code_system.relationships:
            connected_entities.add(rel.source_code_id)
            connected_entities.add(rel.target_code_id)
            
        # Find orphaned entities
        for code_id, code in self.code_system.codes.items():
            if code_id not in connected_entities:
                # Skip actors and enumerations
                if "actor" in code.stereotypes or code.is_enumeration:
                    continue
                    
                self.warnings.append(f"Orphaned entity: '{code.name}' has no relationships")
    
    def _check_relationship_validity(self) -> None:
        """Check relationship validity."""
        for rel in self.code_system.relationships:
            # Check if both ends exist
            if rel.source_code_id not in self.code_system.codes:
                self.issues.append(
                    f"Relationship {rel.id} references non-existent source entity: {rel.source_code_id}"
                )
                continue
                
            if rel.target_code_id not in self.code_system.codes:
                self.issues.append(
                    f"Relationship {rel.id} references non-existent target entity: {rel.target_code_id}"
                )
                continue
                
            source = self.code_system.codes[rel.source_code_id]
            target = self.code_system.codes[rel.target_code_id]
            
            # Check interface implementation
            if rel.relationship_type == CSLRelationshipType.IMPLEMENTATION:
                if not target.is_interface:
                    self.issues.append(
                        f"Implementation relationship {rel.id} targets non-interface: {target.name}"
                    )
            
            # Check self-relationships
            if rel.source_code_id == rel.target_code_id:
                self.issues.append(
                    f"Self-relationship detected: {source.name} relates to itself"
                )
            
            # Check enumerations in relationships
            if source.is_enumeration:
                if rel.relationship_type not in [CSLRelationshipType.ASSOCIATION, CSLRelationshipType.IS_A]:
                    self.issues.append(
                        f"Enumeration '{source.name}' has invalid relationship type: {rel.relationship_type.value}"
                    )
                    
            if target.is_enumeration:
                if rel.relationship_type not in [CSLRelationshipType.ASSOCIATION, CSLRelationshipType.IS_A]:
                    self.issues.append(
                        f"Enumeration '{target.name}' has invalid relationship type: {rel.relationship_type.value}"
                    )
    
    def _check_inheritance_cycles(self) -> None:
        """Check for inheritance cycles."""
        # Track visited nodes for each inheritance path
        for code_id in self.code_system.codes:
            visited = set()
            if self._has_inheritance_cycle(code_id, visited):
                self.issues.append(
                    f"Inheritance cycle detected involving entity: {self.code_system.codes[code_id].name}"
                )
    
    def _has_inheritance_cycle(self, code_id: str, visited: Set[str]) -> bool:
        """
        Check if a code is part of an inheritance cycle.
        
        Args:
            code_id: ID of the code to check
            visited: Set of visited code IDs
            
        Returns:
            True if cycle detected, False otherwise
        """
        if code_id in visited:
            return True
            
        visited.add(code_id)
        
        # Check all parent relationships
        for rel in self.code_system.relationships:
            if rel.source_code_id == code_id and rel.relationship_type == CSLRelationshipType.IS_A:
                if self._has_inheritance_cycle(rel.target_code_id, visited.copy()):
                    return True
                    
        return False
    
    def _check_interface_implementations(self) -> None:
        """Check interface implementations."""
        for code_id, code in self.code_system.codes.items():
            if code.is_interface:
                # Check if interface has methods
                if not code.methods:
                    self.warnings.append(f"Interface '{code.name}' has no methods")
                
                # Check if interface has implementations
                has_implementation = False
                for rel in self.code_system.relationships:
                    if (rel.target_code_id == code_id and 
                        rel.relationship_type == CSLRelationshipType.IMPLEMENTATION):
                        has_implementation = True
                        break
                        
                if not has_implementation:
                    self.warnings.append(f"Interface '{code.name}' has no implementations")
    
    def _check_relationship_multiplicities(self) -> None:
        """Check relationship multiplicities."""
        for rel in self.code_system.relationships:
            # Skip if entities don't exist
            if (rel.source_code_id not in self.code_system.codes or 
                rel.target_code_id not in self.code_system.codes):
                continue
                
            # Check inheritance and implementation multiplicities
            if rel.relationship_type in [CSLRelationshipType.IS_A, CSLRelationshipType.IMPLEMENTATION]:
                if rel.multiplicity["source"] != "1" or rel.multiplicity["target"] != "1":
                    self.issues.append(
                        f"Invalid multiplicity for {rel.relationship_type.value} relationship: "
                        f"{rel.multiplicity['source']}:{rel.multiplicity['target']}"
                    )
    
    def _apply_fixes(self) -> None:
        """Apply automatic fixes to model issues."""
        self._fix_entity_names()
        self._fix_duplicate_entities()
        self._fix_invalid_relationships()
        self._fix_interface_implementations()
        self._fix_relationship_multiplicities()
        self._fix_inheritance_cycles()
    
    def _fix_entity_names(self) -> None:
        """Fix invalid entity names."""
        for code_id, code in self.code_system.codes.items():
            original_name = code.name
            
            # Fix empty names
            if not code.name.strip():
                code.name = f"Entity_{code_id}"
                self.fixes_applied.append(f"Renamed empty entity to '{code.name}'")
                
            # Fix non-alphanumeric characters
            if not re.match(r'^[a-zA-Z0-9_\s]+$', code.name):
                code.name = re.sub(r'[^a-zA-Z0-9_\s]', '', code.name)
                if not code.name:  # If everything was removed
                    code.name = f"Entity_{code_id}"
                    
                self.fixes_applied.append(
                    f"Removed invalid characters from entity name: '{original_name}' → '{code.name}'"
                )
    
    def _fix_duplicate_entities(self) -> None:
        """Fix duplicate entity names."""
        name_map = {}
        duplicates = []
        
        # Find duplicates
        for code_id, code in self.code_system.codes.items():
            name_lower = code.name.lower()
            
            if name_lower in name_map:
                duplicates.append((code_id, name_lower))
            else:
                name_map[name_lower] = code_id
        
        # Fix duplicates
        for code_id, name_lower in duplicates:
            code = self.code_system.codes[code_id]
            original_name = code.name
            
            # Append a suffix
            duplicate_count = 2
            while code.name.lower() in name_map:
                code.name = f"{original_name}_{duplicate_count}"
                duplicate_count += 1
                
            # Update name map
            name_map[code.name.lower()] = code_id
            
            self.fixes_applied.append(
                f"Renamed duplicate entity: '{original_name}' → '{code.name}'"
            )
    
    def _fix_invalid_relationships(self) -> None:
        """Fix invalid relationships."""
        relationships_to_remove = []
        relationships_to_change = []
        
        for rel in self.code_system.relationships:
            # Remove relationships with non-existent ends
            if (rel.source_code_id not in self.code_system.codes or 
                rel.target_code_id not in self.code_system.codes):
                relationships_to_remove.append(rel)
                continue
                
            source = self.code_system.codes[rel.source_code_id]
            target = self.code_system.codes[rel.target_code_id]
            
            # Fix self-relationships
            if rel.source_code_id == rel.target_code_id:
                relationships_to_remove.append(rel)
                continue
                
            # Fix implementation relationships targeting non-interfaces
            if rel.relationship_type == CSLRelationshipType.IMPLEMENTATION and not target.is_interface:
                relationships_to_change.append((rel, CSLRelationshipType.ASSOCIATION))
                continue
                
            # Fix enumeration relationships
            if source.is_enumeration or target.is_enumeration:
                if rel.relationship_type not in [CSLRelationshipType.ASSOCIATION, CSLRelationshipType.IS_A]:
                    relationships_to_change.append((rel, CSLRelationshipType.ASSOCIATION))
        
        # Apply removals
        for rel in relationships_to_remove:
            if rel in self.code_system.relationships:
                self.code_system.relationships.remove(rel)
                self.fixes_applied.append(f"Removed invalid relationship: {rel.id}")
        
        # Apply changes
        for rel, new_type in relationships_to_change:
            old_type = rel.relationship_type
            rel.relationship_type = new_type
            
            # Update association name if needed
            if new_type == CSLRelationshipType.ASSOCIATION and not rel.association_name:
                rel.association_name = "relates to"
                
            self.fixes_applied.append(
                f"Changed relationship type: {rel.id} from {old_type.value} to {new_type.value}"
            )
    
    def _fix_inheritance_cycles(self) -> None:
        """Fix inheritance cycles."""
        # First, identify all cycles
        cycles = self._find_all_inheritance_cycles()
        
        # Break each cycle by removing or changing the weakest relationship
        for cycle in cycles:
            self._break_inheritance_cycle(cycle)
    
    def _find_all_inheritance_cycles(self) -> List[List[str]]:
        """
        Find all inheritance cycles in the model.
        
        Returns:
            List of cycles, where each cycle is a list of code IDs
        """
        cycles = []
        visited_global = set()
        
        for code_id in self.code_system.codes:
            if code_id in visited_global:
                continue
                
            # DFS to find cycles
            path = []
            visited_local = set()
            
            self._dfs_find_cycle(code_id, path, visited_local, visited_global, cycles)
            
        return cycles
    
    def _dfs_find_cycle(
        self, 
        code_id: str, 
        path: List[str], 
        visited_local: Set[str],
        visited_global: Set[str],
        cycles: List[List[str]]
    ) -> None:
        """
        DFS to find inheritance cycles.
        
        Args:
            code_id: Current code ID
            path: Current path being explored
            visited_local: Set of visited nodes in current path
            visited_global: Set of all visited nodes
            cycles: List to store found cycles
        """
        # Check for cycle
        if code_id in visited_local:
            # Extract cycle from path
            start_idx = path.index(code_id)
            cycle = path[start_idx:] + [code_id]
            cycles.append(cycle)
            return
            
        # Add to current path and visited sets
        visited_local.add(code_id)
        visited_global.add(code_id)
        path.append(code_id)
        
        # Explore children
        for rel in self.code_system.relationships:
            if rel.source_code_id == code_id and rel.relationship_type == CSLRelationshipType.IS_A:
                self._dfs_find_cycle(
                    rel.target_code_id, path, visited_local.copy(), 
                    visited_global, cycles
                )
        
        # Remove from current path
        path.pop()
    
    def _break_inheritance_cycle(self, cycle: List[str]) -> None:
        """
        Break an inheritance cycle.
        
        Args:
            cycle: List of code IDs forming a cycle
        """
        # Find all inheritance relationships in the cycle
        cycle_relationships = []
        
        for i in range(len(cycle) - 1):
            source_id = cycle[i]
            target_id = cycle[i + 1]
            
            for rel in self.code_system.relationships:
                if (rel.source_code_id == source_id and 
                    rel.target_code_id == target_id and 
                    rel.relationship_type == CSLRelationshipType.IS_A):
                    cycle_relationships.append(rel)
        
        if not cycle_relationships:
            return
            
        # Choose relationship to break (lowest confidence)
        rel_to_break = min(cycle_relationships, key=lambda r: r.confidence)
        
        # Change to association
        old_type = rel_to_break.relationship_type
        rel_to_break.relationship_type = CSLRelationshipType.ASSOCIATION
        rel_to_break.association_name = "relates to"
        
        source = self.code_system.codes[rel_to_break.source_code_id]
        target = self.code_system.codes[rel_to_break.target_code_id]
        
        self.fixes_applied.append(
            f"Changed relationship to break inheritance cycle: {source.name} -> {target.name}"
        )
    
    def _fix_interface_implementations(self) -> None:
        """Fix interface implementation relationships."""
        for code_id, code in self.code_system.codes.items():
            if code.is_interface:
                # Add methods if missing
                if not code.methods:
                    method_name = f"do{code.name.capitalize().replace(' ', '')}"
                    code.methods = [{
                        "name": method_name,
                        "signature": "(): void",
                        "visibility": "+"
                    }]
                    
                    self.fixes_applied.append(
                        f"Added method {method_name}() to interface '{code.name}'"
                    )
    
    def _fix_relationship_multiplicities(self) -> None:
        """Fix relationship multiplicities."""
        for rel in self.code_system.relationships:
            # Skip if entities don't exist
            if (rel.source_code_id not in self.code_system.codes or 
                rel.target_code_id not in self.code_system.codes):
                continue
                
            # Fix inheritance and implementation multiplicities
            if rel.relationship_type in [CSLRelationshipType.IS_A, CSLRelationshipType.IMPLEMENTATION]:
                if rel.multiplicity["source"] != "1" or rel.multiplicity["target"] != "1":
                    old_multiplicity = f"{rel.multiplicity['source']}:{rel.multiplicity['target']}"
                    rel.multiplicity = {"source": "1", "target": "1"}
                    
                    self.fixes_applied.append(
                        f"Fixed multiplicity for {rel.relationship_type.value} relationship: "
                        f"{old_multiplicity} → 1:1"
                    )