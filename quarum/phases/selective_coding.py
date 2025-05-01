"""
Selective Coding phase implementation for domain modeling.

This module implements the Selective Coding phase of qualitative analysis,
which focuses on refining the domain model and establishing hierarchies.
"""

import time
import json
from typing import List, Dict, Any, Tuple, Set, Optional
from difflib import SequenceMatcher
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from quarum.core.code import Code
from quarum.core.relationship import CodeRelationship
from quarum.core.enums import CSLRelationshipType
from quarum.phases.phase_utils import BasePhase, PhaseContext, PhaseResult
from quarum.nlp.prompt_builder import PromptBuilder


class SelectiveCodingPhase(BasePhase):
    """
    Implements the Selective Coding phase of domain modeling.
    
    Selective Coding focuses on refining the model by establishing
    hierarchies, enriching entities with domain-specific attributes
    and methods, and pruning excessive or duplicate relationships.
    """
    
    def __init__(self):
        """Initialize the Selective Coding phase."""
        super().__init__("Selective Coding")
        self.common_base_classes = ["Entity", "Resource", "Record", "Data"]
        
    def execute(self, context: PhaseContext) -> PhaseResult:
        """
        Execute the Selective Coding phase.
        
        Args:
            context: The shared context information
            
        Returns:
            Results of the phase execution
        """
        start_time = time.time()
        self.messages = []
        self.metrics = {}
        
        # Track initial counts
        initial_class_count = len(context.code_system.codes)
        initial_relationship_count = len(context.code_system.relationships)
        
        # Calculate relevance scores
        context.code_system.calculate_relevance_scores()
        
        # Fix relationship multiplicities
        self._fix_relationship_multiplicities(context)
        
        # Validate model elements
        self._validate_model_elements(context)
        
        # Enrich class features
        enriched_count = self._enrich_class_features(context)
        
        # Prune excessive relationships
        pruned_count = self._prune_excessive_relationships(context)
        
        # Limit relationships per class
        limited_count = self._limit_relationships_per_class(context, max_relationships=3)
        
        # Ensure interface implementations
        new_implementations = self._ensure_interface_implementations(context)
        
        # Ensure inheritance hierarchies
        new_hierarchies = self._ensure_inheritance_hierarchies(context)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Final counts
        final_class_count = len(context.code_system.codes)
        final_relationship_count = len(context.code_system.relationships)
        
        # Prepare phase result
        result = PhaseResult(
            phase_name=self.name,
            success=True,
            metrics={
                "initial_class_count": initial_class_count,
                "final_class_count": final_class_count,
                "initial_relationship_count": initial_relationship_count,
                "final_relationship_count": final_relationship_count,
                "pruned_relationships": pruned_count,
                "limited_relationships": limited_count,
                "enriched_entities": enriched_count,
                "new_implementations": new_implementations,
                "new_hierarchies": new_hierarchies,
            },
            messages=self.messages,
            artifacts={},
            execution_time=execution_time
        )
        
        self.messages.append(
            f"Selective Coding complete. Model contains {final_class_count} classes "
            f"and {final_relationship_count} relationships."
        )
        
        return result
    
    def _fix_relationship_multiplicities(self, context: PhaseContext) -> None:
        """
        Fix relationship multiplicities based on relationship types.
        
        Args:
            context: The shared context
        """
        for rel in context.code_system.relationships:
            if rel.relationship_type in [CSLRelationshipType.IS_A, CSLRelationshipType.IMPLEMENTATION]:
                rel.set_multiplicity("1", "1")
            elif rel.relationship_type == CSLRelationshipType.IS_PART_OF:
                rel.set_multiplicity("1..*", "1")
            else:
                rel.set_multiplicity("1", "*")
                
        self.messages.append("Fixed relationship multiplicities based on relationship types.")
    
    def _validate_model_elements(self, context: PhaseContext) -> None:
        """
        Validate model elements for consistency and correctness.
        
        Args:
            context: The shared context
        """
        self._remove_duplicate_relationships(context)
        self._validate_interfaces(context)
        self._validate_enumerations(context)
        self._validate_inheritance(context)
        self._fix_bidirectional_relationships(context)
        
        self.messages.append("Validated model elements for consistency.")
    
    def _remove_duplicate_relationships(self, context: PhaseContext) -> int:
        """
        Remove duplicate relationships between the same entities.
        
        Args:
            context: The shared context
            
        Returns:
            Number of relationships removed
        """
        # Map to track relationships by source and target
        relationship_map = {}
        for rel in context.code_system.relationships:
            key = (rel.source_code_id, rel.target_code_id)
            if key not in relationship_map:
                relationship_map[key] = []
            relationship_map[key].append(rel)
        
        # Keep only the highest priority relationship for each pair
        to_keep = []
        removed = 0
        
        for key, relations in relationship_map.items():
            if len(relations) <= 1:
                # No duplicates, keep all
                to_keep.extend(relations)
                continue
                
            # Sort by relationship type priority and confidence
            relations.sort(
                key=lambda r: (self._relationship_type_priority(r.relationship_type), r.confidence),
                reverse=True
            )
            
            # Keep the highest priority relationship
            to_keep.append(relations[0])
            removed += len(relations) - 1
        
        # Update the relationships list
        context.code_system.relationships = to_keep
        
        # Update outgoing and incoming relationships
        for code in context.code_system.codes.values():
            code.outgoing_relationships = [r for r in code.outgoing_relationships if r in to_keep]
            code.incoming_relationships = [r for r in code.incoming_relationships if r in to_keep]
            
        self.messages.append(f"Removed {removed} duplicate relationships.")
        return removed
    
    def _relationship_type_priority(self, rel_type: CSLRelationshipType) -> int:
        """
        Get priority for relationship type for sorting.
        
        Args:
            rel_type: Relationship type
            
        Returns:
            Priority value (higher is more important)
        """
        priorities = {
            CSLRelationshipType.IS_A: 10,
            CSLRelationshipType.IMPLEMENTATION: 9,
            CSLRelationshipType.IS_PART_OF: 8,
            CSLRelationshipType.CREATES: 5,
            CSLRelationshipType.MANAGES: 5,
            CSLRelationshipType.USES: 3,
            CSLRelationshipType.DEPENDS_ON: 3,
            CSLRelationshipType.PERFORMS: 2,
            CSLRelationshipType.ASSOCIATION: 1,
        }
        return priorities.get(rel_type, 0)
    
    def _validate_interfaces(self, context: PhaseContext) -> None:
        """
        Validate interface correctness and fix inconsistencies.
        
        Args:
            context: The shared context
        """
        # Fix interface properties
        for code in context.code_system.codes.values():
            if code.is_interface:
                # Interfaces should not have attributes
                code.attributes = []
                
                # Ensure interface stereotype
                if "interface" not in code.stereotypes:
                    code.stereotypes.append("interface")
                    
                # Generate basic methods if none exist
                if not code.methods:
                    code.methods = self._generate_basic_methods(code.name, is_interface=True)
        
        # Fix implementation relationships
        to_fix = []
        for rel in context.code_system.relationships:
            if rel.relationship_type != CSLRelationshipType.IMPLEMENTATION:
                continue
                
            source = context.code_system.codes.get(rel.source_code_id)
            target = context.code_system.codes.get(rel.target_code_id)
            
            if not source or not target:
                # Remove if entities don't exist
                to_fix.append((rel, None))
                continue
                
            if not target.is_interface:
                # Change to association if target is not interface
                to_fix.append((rel, CSLRelationshipType.ASSOCIATION))
                continue
                
            if "actor" in source.stereotypes:
                # Change to association if source is actor
                to_fix.append((rel, CSLRelationshipType.ASSOCIATION))
                continue
        
        # Apply fixes
        for rel, new_type in to_fix:
            if new_type is None:
                if rel in context.code_system.relationships:
                    context.code_system.relationships.remove(rel)
            else:
                rel.relationship_type = new_type
                rel.association_name = "uses"
                rel.set_multiplicity("1", "1")
                
        self.messages.append("Validated interface relationships.")
    
    def _validate_enumerations(self, context: PhaseContext) -> None:
        """
        Validate enumeration correctness and generate values if needed.
        
        Args:
            context: The shared context
        """
        for code in context.code_system.codes.values():
            if code.is_enumeration:
                # Enums should not have attributes or methods
                code.attributes = []
                code.methods = []
                
                # Ensure enumeration stereotype
                if "enumeration" not in code.stereotypes:
                    code.stereotypes.append("enumeration")
                    
                # Generate enum values if none exist
                if not code.enum_values:
                    if code.name in context.code_system.enumerations:
                        code.enum_values = context.code_system.enumerations[code.name]
                    else:
                        code.enum_values = self._generate_enum_values(code.name, context)
                        code.notes.append("Enum values inferred from name")
                        
        self.messages.append("Validated enumerations.")
    
    def _generate_enum_values(self, enum_name: str, context: PhaseContext) -> List[str]:
        """
        Generate reasonable enum values based on name.
        
        Args:
            enum_name: Name of the enumeration
            context: The shared context
            
        Returns:
            List of generated enum values
        """
        common_enums = {
            "status": ["ACTIVE", "INACTIVE", "PENDING", "COMPLETED", "CANCELED"],
            "type": ["STANDARD", "PREMIUM", "BASIC", "CUSTOM"],
            "role": ["ADMIN", "USER", "GUEST", "MODERATOR"],
            "priority": ["HIGH", "MEDIUM", "LOW"],
            "state": ["NEW", "IN_PROGRESS", "DONE", "CANCELLED"],
        }
        
        # Check if enum name contains any common patterns
        name_lower = enum_name.lower()
        for pattern, values in common_enums.items():
            if pattern in name_lower:
                return values
        
        # If no common pattern found, use LLM to generate values
        try:
            llm = ChatOpenAI(
                model=context.model_name,
                openai_api_key=context.api_key,
                temperature=0
            )
            
            prompt = f"""Generate 3-6 enum values for '{enum_name}' in {context.domain_name} domain.
            Return JSON array: ["VALUE1", "VALUE2", "VALUE3"]
            Use UPPER_CASE convention."""
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a domain modeling expert."),
                ("human", prompt)
            ])
            chain = chat_prompt | llm
            
            response = chain.invoke({})
            content = response.content if hasattr(response, 'content') else str(response)
            
            result = self._extract_json_from_response(content)
            if result and isinstance(result, list):
                return result
                
        except Exception as e:
            self.messages.append(f"Error generating enum values: {str(e)}")
            
        # Fallback values
        return ["VALUE1", "VALUE2", "VALUE3"]
    
    def _validate_inheritance(self, context: PhaseContext) -> None:
        """
        Validate inheritance relationships and fix inconsistencies.
        
        Args:
            context: The shared context
        """
        to_fix = []
        for rel in context.code_system.relationships:
            if rel.relationship_type != CSLRelationshipType.IS_A:
                continue
                
            source = context.code_system.codes.get(rel.source_code_id)
            target = context.code_system.codes.get(rel.target_code_id)
            
            if not source or not target:
                # Remove if entities don't exist
                to_fix.append((rel, None))
                continue
                
            if "actor" in source.stereotypes or "actor" in target.stereotypes:
                # Change to association if actor involved
                to_fix.append((rel, CSLRelationshipType.ASSOCIATION))
                continue
                
            if source.is_interface or target.is_interface:
                # Change to implementation if interface involved
                to_fix.append((rel, CSLRelationshipType.IMPLEMENTATION))
                
            elif source.is_enumeration or target.is_enumeration:
                # Change to association if enumeration involved
                to_fix.append((rel, CSLRelationshipType.ASSOCIATION))
                
            elif self._check_inheritance_cycle(source.id, target.id, context):
                # Change to association if cycle detected
                to_fix.append((rel, CSLRelationshipType.ASSOCIATION))
        
        # Apply fixes
        for rel, new_type in to_fix:
            if new_type is None:
                if rel in context.code_system.relationships:
                    context.code_system.relationships.remove(rel)
            else:
                rel.relationship_type = new_type
                rel.association_name = "relates to" if new_type == CSLRelationshipType.ASSOCIATION else rel.association_name
                rel.set_multiplicity("1", "1")
                
        self.messages.append("Validated inheritance relationships.")
    
    def _check_inheritance_cycle(
        self, 
        source_id: str, 
        target_id: str, 
        context: PhaseContext,
        visited: Optional[Set[str]] = None
    ) -> bool:
        """
        Check for inheritance cycles.
        
        Args:
            source_id: Source code ID
            target_id: Target code ID
            context: The shared context
            visited: Set of visited node IDs
            
        Returns:
            True if cycle detected, False otherwise
        """
        if visited is None:
            visited = set()
            
        if source_id in visited:
            return True
            
        visited.add(source_id)
        
        for rel in context.code_system.relationships:
            if rel.relationship_type == CSLRelationshipType.IS_A and rel.source_code_id == target_id:
                if rel.target_code_id == source_id:
                    return True
                if self._check_inheritance_cycle(source_id, rel.target_code_id, context, visited):
                    return True
                    
        return False
    
    def _fix_bidirectional_relationships(self, context: PhaseContext) -> None:
        """
        Fix bidirectional relationships between the same entities.
        
        Args:
            context: The shared context
        """
        relationship_pairs = {}
        for rel in context.code_system.relationships:
            key = tuple(sorted([rel.source_code_id, rel.target_code_id]))
            if key not in relationship_pairs:
                relationship_pairs[key] = []
            relationship_pairs[key].append(rel)
        
        for pair, relations in relationship_pairs.items():
            if len(relations) <= 1:
                continue
                
            # Sort by priority
            relations.sort(
                key=lambda r: (self._relationship_type_priority(r.relationship_type), r.confidence),
                reverse=True
            )
            
            to_keep = relations[0]
            to_remove = relations[1:]
            
            # Try to preserve association name if present
            if to_keep.relationship_type in [CSLRelationshipType.ASSOCIATION, CSLRelationshipType.USES]:
                if not to_keep.association_name:
                    for r in to_remove:
                        if r.association_name:
                            to_keep.association_name = r.association_name
                            break
            
            # Remove redundant relationships
            for rel in to_remove:
                if rel in context.code_system.relationships:
                    context.code_system.relationships.remove(rel)
                    
        self.messages.append("Fixed bidirectional relationships.")
    
    def _enrich_class_features(self, context: PhaseContext) -> int:
        """
        Enrich class features with attributes and methods.
        
        Args:
            context: The shared context
            
        Returns:
            Number of classes enriched
        """
        # Find classes to enrich
        classes_to_enrich = []
        
        for cid, code in context.code_system.codes.items():
            if code.is_enumeration:
                continue
                
            if code.is_interface:
                if len(code.methods) < 2:
                    classes_to_enrich.append((cid, code))
            else:
                if len(code.attributes) < 1 or len(code.methods) < 1:
                    classes_to_enrich.append((cid, code))
        
        # Process in batches
        batch_size = 5
        enriched_count = 0
        
        for i in range(0, len(classes_to_enrich), batch_size):
            batch = classes_to_enrich[i:i+batch_size]
            enriched = self._enrich_class_batch(context, batch)
            enriched_count += enriched
            
        self.messages.append(f"Enriched {enriched_count} classes with attributes and methods.")
        return enriched_count
    
    def _enrich_class_batch(
        self, 
        context: PhaseContext, 
        class_batch: List[Tuple[str, Code]]
    ) -> int:
        """
        Enrich a batch of classes with attributes and methods.
        
        Args:
            context: The shared context
            class_batch: List of (id, code) tuples to enrich
            
        Returns:
            Number of classes enriched
        """
        if not class_batch:
            return 0
            
        # Initialize LLM and prompt builder
        llm = ChatOpenAI(
            model=context.model_name,
            openai_api_key=context.api_key,
            temperature=0.2  # Slightly higher for more varied suggestions
        )
        prompt_builder = PromptBuilder(context.domain_name)
        
        # Prepare class information
        class_info = []
        for cid, code in class_batch:
            info = {
                "id": cid,
                "name": code.name,
                "definition": code.definition,
                "type": "Interface" if code.is_interface else "AbstractClass" if code.is_abstract else "Class",
                "currentAttributes": [f"{a['visibility']}{a['name']}: {a['type']}" for a in code.attributes],
                "currentMethods": [f"{m['visibility']}{m['name']}{m['signature']}" for m in code.methods],
            }
            class_info.append(info)
        
        # Create the prompt
        prompt = prompt_builder.build_prompt(
            "class_enrichment",
            class_info_json=json.dumps(class_info, indent=2)
        )
        
        # Gather relevant context
        context_docs = []
        for cid, code in class_batch:
            docs = self.get_evidence_from_query(
                f"{context.domain_name} {code.name} attributes methods",
                context
            )
            if docs:
                context_docs.extend(docs)
                
        # Add context to prompt if available
        if context_docs:
            context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])
            prompt = prompt_builder.add_context_to_prompt(prompt, context_text)
            
        # Create and invoke the chain
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_builder.system_prompt),
            ("human", prompt)
        ])
        chain = chat_prompt | llm
        
        enriched_count = 0
        
        try:
            response = chain.invoke({})
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON result
            result = self._extract_json_from_response(content)
            
            if result:
                for enriched in result.get("enrichedClasses", []):
                    class_id = enriched.get("id")
                    
                    if not class_id or class_id not in context.code_system.codes:
                        continue
                        
                    code_obj = context.code_system.codes[class_id]
                    
                    # Add attributes for non-interfaces
                    if not code_obj.is_interface:
                        attributes = [{
                            "name": a.get("name", ""),
                            "type": a.get("type", "String"),
                            "visibility": a.get("visibility", "+"),
                        } for a in enriched.get("attributes", [])]
                        
                        if attributes:
                            code_obj.attributes = attributes
                            code_obj.notes.append("Attributes enriched based on domain analysis")
                            enriched_count += 1
                    
                    # Add methods for all classes
                    operations = [{
                        "name": o.get("name", ""),
                        "signature": o.get("signature", "(): void"),
                        "visibility": o.get("visibility", "+"),
                    } for o in enriched.get("operations", [])]
                    
                    if operations:
                        code_obj.methods = operations
                        code_obj.notes.append("Methods enriched based on domain analysis")
                        enriched_count += 1
            
        except Exception as e:
            self.messages.append(f"Error enriching classes: {str(e)}")
            
        # Generate basic features if LLM call failed
        if not result:
            for cid, code in class_batch:
                enriched = False
                
                if not code.is_interface and not code.attributes:
                    code.attributes = self._generate_basic_attributes(code.name)
                    code.notes.append("Basic attributes generated")
                    enriched = True
                    
                if not code.methods:
                    code.methods = self._generate_basic_methods(code.name, code.is_interface)
                    code.notes.append("Basic methods generated")
                    enriched = True
                    
                if enriched:
                    enriched_count += 1
        
        return enriched_count
    
    def _generate_basic_attributes(self, class_name: str) -> List[Dict[str, str]]:
        """
        Generate basic attributes for a class based on its name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of attribute dictionaries
        """
        attributes = [
            {"name": "id", "type": "String", "visibility": "+"},
            {"name": "name", "type": "String", "visibility": "+"},
        ]
        
        name_lower = class_name.lower()
        
        if "date" in name_lower:
            attributes.append({"name": "date", "type": "Date", "visibility": "+"})
            
        if "user" in name_lower or "account" in name_lower:
            attributes.append({"name": "email", "type": "String", "visibility": "-"})
            
        return attributes
    
    def _generate_basic_methods(self, class_name: str, is_interface: bool = False) -> List[Dict[str, str]]:
        """
        Generate basic methods for a class based on its name.
        
        Args:
            class_name: Name of the class
            is_interface: Whether the class is an interface
            
        Returns:
            List of method dictionaries
        """
        methods = [
            {"name": "getId", "signature": "(): String", "visibility": "+"},
            {"name": "getName", "signature": "(): String", "visibility": "+"},
        ]
        
        name_lower = class_name.lower()
        
        if name_lower.endswith("repository"):
            entity = class_name[:-10]
            methods.extend([
                {"name": f"find{entity}ById", "signature": "(id: String): "+entity, "visibility": "+"},
                {"name": "findAll", "signature": f"(): List<{entity}>", "visibility": "+"},
            ])
            
        elif name_lower.endswith("service"):
            entity = class_name[:-7]
            methods.extend([
                {"name": f"create{entity}", "signature": f"({entity.lower()}: {entity}): {entity}", "visibility": "+"},
                {"name": f"update{entity}", "signature": f"({entity.lower()}: {entity}): Boolean", "visibility": "+"},
            ])
            
        return methods
    
    def _prune_excessive_relationships(self, context: PhaseContext) -> int:
        """
        Prune excessive relationships between entities.
        
        Args:
            context: The shared context
            
        Returns:
            Number of relationships pruned
        """
        seen_pairs = set()
        to_remove = []
        
        # Sort relationships by priority
        sorted_relationships = sorted(
            context.code_system.relationships,
            key=lambda r: (self._relationship_type_priority(r.relationship_type), r.confidence),
            reverse=True
        )
        
        # Keep only highest priority relationship between any pair
        for rel in sorted_relationships:
            pair = tuple(sorted([rel.source_code_id, rel.target_code_id]))
            
            if pair in seen_pairs:
                to_remove.append(rel)
            else:
                seen_pairs.add(pair)
        
        # Remove pruned relationships
        for rel in to_remove:
            if rel in context.code_system.relationships:
                context.code_system.relationships.remove(rel)
                
        self.messages.append(f"Pruned {len(to_remove)} excessive relationships.")
        return len(to_remove)
    
    def _limit_relationships_per_class(self, context: PhaseContext, max_relationships: int = 3) -> int:
        """
        Limit the number of relationships per class.
        
        Args:
            context: The shared context
            max_relationships: Maximum number of relationships per class
            
        Returns:
            Number of relationships removed
        """
        # Count outgoing relationships per class
        class_rel_counts = {}
        
        for rel in sorted(
            context.code_system.relationships,
            key=lambda r: (self._relationship_type_priority(r.relationship_type), r.confidence),
            reverse=True
        ):
            source_id = rel.source_code_id
            class_rel_counts[source_id] = class_rel_counts.get(source_id, 0) + 1
        
        # Identify relationships to remove
        to_remove = []
        
        for rel in context.code_system.relationships:
            if class_rel_counts[rel.source_code_id] > max_relationships:
                to_remove.append(rel)
                class_rel_counts[rel.source_code_id] -= 1
        
        # Remove relationships
        for rel in to_remove:
            if rel in context.code_system.relationships:
                context.code_system.relationships.remove(rel)
                
        self.messages.append(f"Limited relationships: removed {len(to_remove)} relationships.")
        return len(to_remove)
    
    def _ensure_interface_implementations(self, context: PhaseContext) -> int:
        """
        Ensure interfaces have implementations.
        
        Args:
            context: The shared context
            
        Returns:
            Number of new implementations added
        """
        # Get interfaces
        interfaces = [code for _, code in context.code_system.codes.items() if code.is_interface]
        added_count = 0
        
        for interface in interfaces:
            # Check if interface already has implementations
            has_implementation = False
            
            for rel in context.code_system.relationships:
                if (rel.target_code_id == interface.id and 
                    rel.relationship_type == CSLRelationshipType.IMPLEMENTATION):
                    has_implementation = True
                    break
                    
            if has_implementation:
                continue
                
            # Find implementation candidates
            implementer_candidates = []
            
            for cid, code in context.code_system.codes.items():
                # Skip if not suitable for implementation
                if (code.is_interface or code.is_enumeration or 
                    "actor" in code.stereotypes):
                    continue
                    
                # Calculate name similarity
                similarity = SequenceMatcher(
                    None, interface.name.lower(), code.name.lower()
                ).ratio()
                
                # Score candidates
                if similarity > 0.8:
                    implementer_candidates.append((cid, 3))
                elif similarity > 0.5:
                    implementer_candidates.append((cid, 2))
                else:
                    implementer_candidates.append((cid, 1))
            
            # Add implementation relationship if candidates exist
            if implementer_candidates:
                implementer_candidates.sort(key=lambda x: x[1], reverse=True)
                implementer_id = implementer_candidates[0][0]
                
                rel_id = f"rel_{len(context.code_system.relationships)+1}"
                new_rel = CodeRelationship(
                    relationship_id=rel_id,
                    source_code_id=implementer_id,
                    target_code_id=interface.id,
                    relationship_type=CSLRelationshipType.IMPLEMENTATION,
                    association_name="implements",
                    confidence=0.7,
                    multiplicity={"source": "1", "target": "1"},
                )
                
                if context.code_system.add_relationship(new_rel):
                    context.code_system.codes[implementer_id].notes.append(
                        f"Added implementation of {interface.name}"
                    )
                    added_count += 1
        
        self.messages.append(f"Added {added_count} interface implementations.")
        return added_count
    
    def _ensure_inheritance_hierarchies(self, context: PhaseContext) -> int:
        """
        Ensure appropriate inheritance hierarchies.
        
        Args:
            context: The shared context
            
        Returns:
            Number of new inheritance relationships added
        """
        # Base classes to consider
        common_base_classes = self.common_base_classes.copy()
        
        # Add actors as potential base classes
        for actor in context.code_system.domain_actors:
            if actor not in common_base_classes:
                common_base_classes.append(actor)
        
        # Track codes that already have parents
        has_parent = set()
        for rel in context.code_system.relationships:
            if rel.relationship_type == CSLRelationshipType.IS_A:
                has_parent.add(rel.source_code_id)
        
        added_count = 0
        
        # Find parent candidates for each code
        for cid, code in context.code_system.codes.items():
            # Skip if already has parent or not suitable
            if (code.is_enumeration or code.is_interface or 
                "actor" in code.stereotypes or cid in has_parent):
                continue
                
            candidates = []
            
            # Evaluate potential parents
            for parent_id, parent in context.code_system.codes.items():
                if (parent_id == cid or parent.is_interface or 
                    parent.is_enumeration or "actor" in parent.stereotypes):
                    continue
                    
                # Calculate compatibility score
                score = self._inheritance_compatibility_score(code, parent, common_base_classes)
                
                if score >= 2:
                    candidates.append((parent_id, score))
            
            # Add inheritance relationship if candidates exist
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_parent_id = candidates[0][0]
                
                rel_id = f"rel_{len(context.code_system.relationships)+1}"
                new_rel = CodeRelationship(
                    relationship_id=rel_id,
                    source_code_id=cid,
                    target_code_id=best_parent_id,
                    relationship_type=CSLRelationshipType.IS_A,
                    association_name="extends",
                    confidence=0.7,
                    multiplicity={"source": "1", "target": "1"},
                )
                
                if context.code_system.add_relationship(new_rel):
                    code.notes.append(
                        f"Added inheritance from {context.code_system.codes[best_parent_id].name}"
                    )
                    added_count += 1
        
        self.messages.append(f"Added {added_count} inheritance relationships.")
        return added_count
    
    def _inheritance_compatibility_score(
        self, 
        child: Code, 
        parent: Code,
        common_base_classes: List[str]
    ) -> float:
        """
        Calculate inheritance compatibility score between classes.
        
        Args:
            child: Child code
            parent: Potential parent code
            common_base_classes: List of common base class names
            
        Returns:
            Compatibility score (higher means more compatible)
        """
        score = 0.0
        
        # Name similarity
        child_name = child.name.lower()
        parent_name = parent.name.lower()
        similarity = SequenceMatcher(None, child_name, parent_name).ratio()
        score += similarity * 3
        
        # Child name contains parent name
        if parent_name in child_name and parent_name != child_name:
            score += 2
        
        # Common attributes
        child_attrs = set(a["name"] for a in child.attributes)
        parent_attrs = set(a["name"] for a in parent.attributes)
        common_attrs = len(child_attrs.intersection(parent_attrs))
        score += common_attrs * 0.5
        
        # Abstract classes get bonus
        if parent.is_abstract:
            score += 1
        
        # Common base classes get bonus
        if parent_name in [n.lower() for n in common_base_classes]:
            score += 1
            
        return score