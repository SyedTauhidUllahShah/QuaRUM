"""
Axial Coding phase implementation for domain modeling.

This module implements the Axial Coding phase of qualitative analysis,
which focuses on identifying relationships between domain entities.
"""

import time
import json
from typing import List, Dict, Any, Tuple, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from quarum.core.enums import CSLRelationshipType
from quarum.core.relationship import CodeRelationship
from quarum.phases.phase_utils import BasePhase, PhaseContext, PhaseResult
from quarum.nlp.prompt_builder import PromptBuilder


class AxialCodingPhase(BasePhase):
    """
    Implements the Axial Coding phase of domain modeling.
    
    Axial Coding focuses on identifying relationships between the entities
    discovered during Open Coding. This phase analyzes entity pairs to
    determine the appropriate relationship types and properties.
    """
    
    def __init__(self):
        """Initialize the Axial Coding phase."""
        super().__init__("Axial Coding")
        
    def execute(self, context: PhaseContext) -> PhaseResult:
        """
        Execute the Axial Coding phase.
        
        Args:
            context: The shared context information
            
        Returns:
            Results of the phase execution
        """
        start_time = time.time()
        self.messages = []
        self.metrics = {}
        
        # Initialize LLM and prompt builder
        llm = ChatOpenAI(
            model=context.model_name,
            openai_api_key=context.api_key,
            temperature=0
        )
        prompt_builder = PromptBuilder(context.domain_name)
        
        # First, identify interfaces and abstract classes
        self._identify_interfaces_and_abstracts(context, llm, prompt_builder)
        
        # Then, identify relationships between entities
        relationship_count = self._identify_relationships(context, llm, prompt_builder)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare phase result
        result = PhaseResult(
            phase_name=self.name,
            success=True,
            metrics={
                "relationship_count": relationship_count,
                "interface_count": len([c for c in context.code_system.codes.values() if c.is_interface]),
                "abstract_class_count": len([c for c in context.code_system.codes.values() if c.is_abstract]),
            },
            messages=self.messages,
            artifacts={},
            execution_time=execution_time
        )
        
        self.messages.append(f"Axial Coding complete. Established {relationship_count} relationships.")
        
        return result
    
    def _identify_interfaces_and_abstracts(
        self, 
        context: PhaseContext,
        llm: Any,
        prompt_builder: PromptBuilder
    ) -> None:
        """
        Identify interfaces and abstract classes.
        
        Args:
            context: The shared context
            llm: Language model instance
            prompt_builder: Prompt builder instance
        """
        # Filter valid codes
        valid_codes = []
        for cid, code in context.code_system.codes.items():
            if not self._is_banned_term(code.name, context.metadata["banned_words"]):
                valid_codes.append({
                    "id": cid,
                    "name": code.name,
                    "methods": [m["signature"] for m in code.methods],
                    "attributes": [f"{a['name']}: {a['type']}" for a in code.attributes],
                    "stereotypes": code.stereotypes,
                    "extracted_text": code.extracted_text
                })
        
        if not valid_codes:
            self.messages.append("No valid codes found for interface identification.")
            return
            
        # Create the prompt
        prompt = prompt_builder.build_prompt(
            "interface_identification",
            classes_json=json.dumps(valid_codes[:20], indent=2)
        )
        
        # Create and invoke the chain
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_builder.system_prompt),
            ("human", prompt)
        ])
        chain = chat_prompt | llm
        
        try:
            response = chain.invoke({})
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON result
            result = self._extract_json_from_response(content)
            if not result:
                self.messages.append("Failed to extract interface identification results.")
                return
                
            # Process interfaces
            interface_count = 0
            for interface in result.get("interfaces", []):
                class_id = interface.get("id", "")
                evidence = interface.get("evidence", "")
                
                if not class_id or not evidence or class_id not in context.code_system.codes:
                    continue
                    
                code = context.code_system.codes[class_id]
                if evidence.lower() in code.extracted_text.lower():
                    code.is_interface = True
                    if "interface" not in code.stereotypes:
                        code.stereotypes.append("interface")
                    interface_count += 1
            
            # Process abstract classes
            abstract_count = 0
            for abstract in result.get("abstractClasses", []):
                class_id = abstract.get("id", "")
                evidence = abstract.get("evidence", "")
                
                if not class_id or not evidence or class_id not in context.code_system.codes:
                    continue
                    
                code = context.code_system.codes[class_id]
                if evidence.lower() in code.extracted_text.lower():
                    code.is_abstract = True
                    if "abstract" not in code.stereotypes:
                        code.stereotypes.append("abstract")
                    abstract_count += 1
                    
            self.messages.append(f"Identified {interface_count} interfaces and {abstract_count} abstract classes.")
            
        except Exception as e:
            self.messages.append(f"Error identifying interfaces and abstracts: {str(e)}")
    
    def _identify_relationships(
        self, 
        context: PhaseContext,
        llm: Any,
        prompt_builder: PromptBuilder
    ) -> int:
        """
        Identify relationships between entities.
        
        Args:
            context: The shared context
            llm: Language model instance
            prompt_builder: Prompt builder instance
            
        Returns:
            Number of relationships identified
        """
        # Get valid classes
        valid_classes = [
            c for cid, c in context.code_system.codes.items() 
            if not self._is_banned_term(c.name, context.metadata["banned_words"])
        ]
        
        # Create pairs to analyze
        pairs_to_analyze = []
        for i, c1 in enumerate(valid_classes):
            for c2 in valid_classes[i+1:]:
                pairs_to_analyze.append((c1.id, c2.id))
                
        self.messages.append(f"Analyzing {len(pairs_to_analyze)} potential entity relationships...")
        
        # Process in batches
        batch_size = 5
        relationship_count = 0
        
        for i in range(0, len(pairs_to_analyze), batch_size):
            batch = pairs_to_analyze[i:i+batch_size]
            rel_count = self._identify_relationship_batch(context, batch, llm, prompt_builder)
            relationship_count += rel_count
            
            if (i + batch_size) % 20 == 0 or (i + batch_size) >= len(pairs_to_analyze):
                self.messages.append(f"Analyzed {i + len(batch)}/{len(pairs_to_analyze)} pairs, found {relationship_count} relationships.")
                
        return relationship_count
    
    def _identify_relationship_batch(
        self, 
        context: PhaseContext, 
        id_pairs: List[Tuple[str, str]],
        llm: Any,
        prompt_builder: PromptBuilder
    ) -> int:
        """
        Identify relationships for a batch of entity pairs.
        
        Args:
            context: The shared context
            id_pairs: List of entity ID pairs to analyze
            llm: Language model instance
            prompt_builder: Prompt builder instance
            
        Returns:
            Number of relationships identified in this batch
        """
        # Prepare pair information
        pair_info = []
        for source_id, target_id in id_pairs:
            source = context.code_system.codes[source_id]
            target = context.code_system.codes[target_id]
            
            pair_info.append({
                "source": {
                    "id": source_id,
                    "name": source.name,
                    "extracted_text": source.extracted_text
                },
                "target": {
                    "id": target_id,
                    "name": target.name,
                    "extracted_text": target.extracted_text
                }
            })
            
        # Add domain relationship hints to the prompt
        domain_rel_hints = ""
        for pair in pair_info:
            src_name = context.code_system.codes[pair["source"]["id"]].name
            if src_name in context.code_system.domain_relationships:
                domain_rel_hints += f"Domain hint: {src_name} has these relationships: {', '.join(context.code_system.domain_relationships[src_name])}\n"
        
        # Create the prompt
        prompt = prompt_builder.build_prompt(
            "relationship_analysis",
            class_pairs_json=json.dumps(pair_info, indent=2),
            domain_rel_hints=domain_rel_hints
        )
        
        # Gather relevant context chunks
        context_chunks = []
        for source_id, target_id in id_pairs:
            source_name = context.code_system.codes[source_id].name
            target_name = context.code_system.codes[target_id].name
            
            # Find evidence for this relationship
            query = f"{context.domain_name} {source_name} relationship {target_name}"
            docs = self.get_evidence_from_query(query, context)
            
            if docs:
                context_chunks.extend(docs[:2])
                
        # Add context to prompt if available
        if context_chunks:
            context_text = "\n\n".join([doc.page_content for doc in context_chunks[:3]])
            prompt = prompt_builder.add_context_to_prompt(prompt, context_text)
            
        # Create and invoke the chain
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_builder.system_prompt),
            ("human", prompt)
        ])
        chain = chat_prompt | llm
        
        try:
            response = chain.invoke({})
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON result
            result = self._extract_json_from_response(content)
            if not result:
                return 0
                
            # Process relationships
            count = 0
            for rel in result.get("relationships", []):
                rel_type = rel.get("type", "NONE")
                
                # Skip if no relationship
                if rel_type == "NONE":
                    continue
                    
                # Get entity IDs
                source_id = rel.get("sourceId")
                target_id = rel.get("targetId")
                
                # Skip if entities don't exist
                if not source_id in context.code_system.codes or not target_id in context.code_system.codes:
                    continue
                    
                # Get relationship data
                evidence = rel.get("evidence", "")
                confidence = float(rel.get("confidence", 0.6))
                
                # Skip if confidence too low
                if confidence < 0.7:
                    continue
                    
                association_name = rel.get("association_name", "")
                multiplicity = rel.get("multiplicity", {"source": "1", "target": "*"})
                
                # Map relationship type
                relationship_type = CSLRelationshipType.ASSOCIATION
                if rel_type == "IS_A":
                    relationship_type = CSLRelationshipType.IS_A
                elif rel_type == "IS_PART_OF":
                    relationship_type = CSLRelationshipType.IS_PART_OF
                elif rel_type == "IMPLEMENTS":
                    relationship_type = CSLRelationshipType.IMPLEMENTATION
                elif rel_type == "DEPENDS_ON":
                    relationship_type = CSLRelationshipType.DEPENDS_ON
                elif rel_type == "USES":
                    relationship_type = CSLRelationshipType.USES
                elif rel_type == "MANAGES":
                    relationship_type = CSLRelationshipType.MANAGES
                elif rel_type == "CREATES":
                    relationship_type = CSLRelationshipType.CREATES
                
                # Create relationship object
                relationship_id = f"rel_{len(context.code_system.relationships)+1}"
                new_rel = CodeRelationship(
                    id=relationship_id,
                    source_code_id=source_id,
                    target_code_id=target_id,
                    relationship_type=relationship_type,
                    association_name=association_name,
                    confidence=confidence,
                    multiplicity=multiplicity
                )
                
                # Add evidence if available
                if evidence:
                    new_rel.add_evidence(evidence)
                
                # Handle special case for implementation relationships
                source = context.code_system.codes[source_id]
                target = context.code_system.codes[target_id]
                
                if relationship_type == CSLRelationshipType.IMPLEMENTATION and not target.is_interface:
                    new_rel.relationship_type = CSLRelationshipType.ASSOCIATION
                    new_rel.association_name = "uses"
                
                # Add relationship to code system
                if context.code_system.add_relationship(new_rel):
                    count += 1
                    
            return count
            
        except Exception as e:
            self.messages.append(f"Error identifying relationships in batch: {str(e)}")
            return 0