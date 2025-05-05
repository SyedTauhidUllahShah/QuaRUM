"""
Open Coding phase implementation for domain modeling.

This module implements the Open Coding phase of qualitative analysis,
which focuses on identifying and extracting domain entities from text.
"""

import time
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from quarum.core.code import Code
from quarum.phases.phase_utils import BasePhase, PhaseContext, PhaseResult
from quarum.nlp.prompt_builder import PromptBuilder


class OpenCodingPhase(BasePhase):
    """
    Implements the Open Coding phase of domain modeling.
    
    Open Coding focuses on identifying and extracting domain entities
    from text. This phase analyzes each chunk of text to find potential
    classes, interfaces, and enumerations with their properties.
    """
    
    def __init__(self):
        """Initialize the Open Coding phase."""
        super().__init__("Open Coding")
        
    def execute(self, context: PhaseContext) -> PhaseResult:
        """
        Execute the Open Coding phase.
        
        Args:
            context: The shared context information
            
        Returns:
            Results of the phase execution
        """
        start_time = time.time()
        self.messages = []
        self.metrics = {}
        
        # Initialize the prompt builder
        prompt_builder = PromptBuilder(context.domain_name)
        
        # Create LLM instance
        llm = ChatOpenAI(
            model=context.model_name,
            openai_api_key=context.api_key,
            temperature=0
        )
        
        # Track statistics
        extracted_entities = 0
        processed_chunks = 0
        
        # Process each document chunk
        for i, doc in enumerate(context.documents):
            chunk_text = doc.page_content
            
            # Skip if chunk is too small
            if len(chunk_text) < 50:
                continue
                
            self.messages.append(f"Processing chunk {i+1}/{len(context.documents)}...")
            processed_chunks += 1
            
            # Prepare domain context for the prompt
            domain_context = ""
            if context.code_system.core_domain_entities:
                concepts_str = ", ".join(context.code_system.core_domain_entities[:7])
                domain_context = f"\nKey domain concepts: {concepts_str}"
            
            # Build the prompt for entity extraction
            prompt = prompt_builder.build_prompt(
                "entity_extraction",
                text_chunk=chunk_text,
                domain_context=domain_context
            )
            
            # Try entity extraction with retries
            entities = None
            for attempt in range(2):
                # Create and invoke the chain
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_builder.system_prompt),
                    ("human", prompt)
                ])
                chain = chat_prompt | llm
                
                try:
                    response = chain.invoke({})
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Process extracted entities
                    entities = self._process_extracted_entities(
                        content, doc, context.code_system.raw_text, 
                        context.metadata["banned_words"]
                    )
                    
                    if entities:
                        extracted_entities += len(entities)
                        break
                        
                except Exception as e:
                    self.messages.append(f"Error in LLM call: {str(e)}")
                
                self.messages.append(f"Open coding attempt {attempt+1} failed for chunk {i+1}, retrying...")
            
            # If extraction failed, try to add core entities
            if not entities and context.code_system.core_domain_entities:
                self._add_core_entities_from_chunk(context, doc, chunk_text)
        
        # Ensure core entities are present in the model
        self._ensure_core_entities_present(context)
        
        # Filter out invalid entities
        self._filter_entities(context)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare phase result
        result = PhaseResult(
            phase_name=self.name,
            success=True,
            metrics={
                "processed_chunks": processed_chunks,
                "extracted_entities": extracted_entities,
                "final_entity_count": len(context.code_system.codes),
            },
            messages=self.messages,
            artifacts={},
            execution_time=execution_time
        )
        
        self.messages.append(f"Open Coding complete. Extracted {len(context.code_system.codes)} candidate entities.")
        
        return result
    
    def _process_extracted_entities(
        self, 
        response: str, 
        doc: Document, 
        raw_text: str,
        banned_words: set
    ) -> List[Dict[str, Any]]:
        """
        Process entities extracted by the LLM.
        
        Args:
            response: LLM response containing entity data
            doc: Source document chunk
            raw_text: Raw text of the entire corpus
            banned_words: Set of banned generic terms
            
        Returns:
            List of processed entity data dictionaries
        """
        # Extract JSON from response
        result = self._extract_json_from_response(response)
        if not result:
            return []
            
        entities = result.get("entities", [])
        if not entities:
            return []
            
        # Get metadata from document
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source = doc.metadata.get("source", "unknown")
        
        processed_entities = []
        for entity in entities:
            entity_name = entity.get("name", "").strip()
            
            # Skip if invalid name
            if not entity_name or self._is_banned_term(entity_name, banned_words):
                continue
                
            # Check if name appears in text or is a core entity
            name_in_text = self._verify_text_presence(entity_name, raw_text)
            
            # Skip if not in text and not mentioned as core entity
            if not name_in_text:
                continue
                
            # Get confidence and skip if too low
            confidence = float(entity.get("confidence", 0.5))
            if confidence < 0.5:
                continue
                
            # Process entity data
            processed_entities.append(entity)
            
        return processed_entities
    
    def _add_core_entities_from_chunk(
        self, 
        context: PhaseContext, 
        doc: Document, 
        chunk_text: str
    ) -> None:
        """
        Add core domain entities referenced in a chunk.
        
        Args:
            context: The shared context
            doc: Source document
            chunk_text: Text content of the chunk
        """
        for entity_name in context.code_system.core_domain_entities[:2]:
            # Skip if entity already exists
            if any(code.name.lower() == entity_name.lower() 
                  for code in context.code_system.codes.values()):
                continue
                
            # Create a new code object
            new_id = f"code_{len(context.code_system.codes)+1}"
            code_obj = Code(
                code_id=new_id,
                name=entity_name,
                definition=f"Core entity in {context.domain_name} domain",
                is_recommendation=True
            )
            
            # Add attributes if available
            if entity_name in context.code_system.domain_attributes:
                code_obj.attributes = context.code_system.domain_attributes[entity_name]
                
            # Add methods if available
            if entity_name in context.code_system.domain_operations:
                code_obj.methods = context.code_system.domain_operations[entity_name]
                
            # Add stereotype
            if entity_name in context.code_system.domain_actors:
                code_obj.stereotypes = ["actor"]
            else:
                code_obj.stereotypes = ["entity"]
                
            # Set confidence and evidence
            code_obj.confidence = 0.6
            code_obj.evidence_chunks.append(chunk_text[:100])
            code_obj.evidence_locations.append(f"{doc.metadata.get('source')}:{doc.metadata.get('chunk_id')}")
            code_obj.notes.append("Added from domain analysis as core entity")
            
            # Add to code system
            context.code_system.add_code(code_obj)
    
    def _ensure_core_entities_present(self, context: PhaseContext) -> None:
        """
        Ensure all core domain entities are in the model.
        
        Args:
            context: The shared context
        """
        for entity_name in context.code_system.core_domain_entities:
            # Check if entity already exists
            exists = False
            for code in context.code_system.codes.values():
                if code.name.lower() == entity_name.lower():
                    exists = True
                    break
                    
            if exists:
                continue
                
            # Create new code for core entity
            new_id = f"code_{len(context.code_system.codes)+1}"
            new_code = Code(
                code_id=new_id,
                name=entity_name,
                definition=f"Core entity in {context.domain_name} domain",
                is_recommendation=True
            )
            
            # Add attributes if available
            if entity_name in context.code_system.domain_attributes:
                new_code.attributes = context.code_system.domain_attributes[entity_name]
                
            # Add methods if available
            if entity_name in context.code_system.domain_operations:
                new_code.methods = context.code_system.domain_operations[entity_name]
                
            # Add stereotype
            if entity_name in context.code_system.domain_actors:
                new_code.stereotypes = ["actor"]
            else:
                new_code.stereotypes = ["entity"]
                
            # Set confidence
            new_code.confidence = 0.6
            new_code.notes.append("Added from domain analysis as core entity")
            
            # Add to code system
            context.code_system.add_code(new_code)
    
    def _filter_entities(self, context: PhaseContext) -> None:
        """
        Filter out invalid or low-quality entities.
        
        Args:
            context: The shared context
        """
        to_remove = []
        banned_words = context.metadata["banned_words"]
        
        for cid, code in context.code_system.codes.items():
            # Remove if banned term
            if self._is_banned_term(code.name, banned_words):
                to_remove.append(cid)
                continue
                
            # Remove if low confidence and not recommendation
            if code.confidence < 0.4 and not code.is_recommendation:
                to_remove.append(cid)
                continue
                
        # Remove filtered entities
        for rid in to_remove:
            if rid in context.code_system.codes:
                del context.code_system.codes[rid]
                
        self.messages.append(f"Filtered out {len(to_remove)} invalid entities.")