"""
Domain modeling pipeline implementation.

This module implements the main domain modeling pipeline that
orchestrates the entire process of extracting a domain model
from natural language requirements.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

from langchain.docstore.document import Document

from domain_modeler.pipelines.base_pipeline import Pipeline, PipelineResult
from domain_modeler.phases.phase_utils import PhaseContext
from domain_modeler.phases.open_coding import OpenCodingPhase
from domain_modeler.phases.axial_coding import AxialCodingPhase
from domain_modeler.phases.selective_coding import SelectiveCodingPhase
from domain_modeler.generators.plantuml import PlantUMLGenerator
from domain_modeler.generators.report import ReportGenerator
from domain_modeler.core.code_system import CodeSystem
from domain_modeler.nlp.embeddings import VectorEmbeddings
from domain_modeler.nlp.chunking import TextChunker
from domain_modeler.utils.llm_client import LLMClient
from domain_modeler.utils.validation import ModelValidator
from domain_modeler.config.settings import Settings

logger = logging.getLogger(__name__)

class DomainModelingPipeline(Pipeline):
    """
    Pipeline for domain model extraction from natural language requirements.
    
    This pipeline orchestrates the entire process of domain modeling:
    1. Analyzing domain description
    2. Processing requirements text
    3. Extracting entities (Open Coding)
    4. Identifying relationships (Axial Coding)
    5. Refining the model (Selective Coding)
    6. Generating UML diagrams and reports
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-mini"):
        """
        Initialize the domain modeling pipeline.
        
        Args:
            api_key: OpenAI API key
            model_name: LLM model name
        """
        super().__init__("Domain Modeling")
        self.api_key = api_key
        self.model_name = model_name
        self.domain_name = ""
        self.domain_description = ""
        self.code_system = CodeSystem()
        self.llm_client = None
        self.vector_embeddings = None
        self.documents = []
        self.settings = Settings()
        
        # Initialize with default settings
        self._setup_components()
    
    def _setup_components(self) -> None:
        """Set up pipeline components."""
        # Initialize LLM client
        self.llm_client = LLMClient(
            api_key=self.api_key,
            model_name=self.model_name,
            max_retries=self.settings.get("llm", "max_retries"),
            retry_delay=self.settings.get("llm", "retry_delay"),
            temperature=self.settings.get("llm", "temperature")
        )
        
        # Initialize vector embeddings
        self.vector_embeddings = VectorEmbeddings(
            api_key=self.api_key
        )
    
    def setup(
        self, 
        file_path: str, 
        domain_description: str,
        domain_name: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> bool:
        """
        Set up the pipeline with input files and description.
        
        Args:
            file_path: Path to the requirements file
            domain_description: Description of the domain
            domain_name: Optional name of the domain
            settings: Optional custom settings
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Update settings if provided
            if settings:
                self.settings = settings
                self._setup_components()
            
            # Store domain information
            self.domain_description = domain_description
            self.code_system.domain_description = domain_description
            
            # Extract domain name from description or use provided name
            if domain_name:
                self.domain_name = domain_name
            else:
                self.domain_name = self._extract_domain_name(domain_description)
                
            self.add_message(f"Setting up pipeline for domain: {self.domain_name}")
            
            # Load and process the requirements file
            if not self._load_document(file_path):
                return False
                
            # Analyze domain to extract core concepts
            if not self._analyze_domain():
                return False
                
            self.add_message(f"Setup complete. Core entities: {', '.join(self.code_system.core_domain_entities[:5])}")
            self.add_message(f"Loaded {len(self.documents)} document chunks")
            
            return True
            
        except Exception as e:
            self.add_error(f"Setup failed: {str(e)}")
            return False
    
    def _extract_domain_name(self, description: str) -> str:
        """
        Extract domain name from description.
        
        Args:
            description: Domain description
            
        Returns:
            Extracted domain name
        """
        import re
        match = re.search(r"([A-Za-z\s]+)(?:\s+system|\s+domain)?", description)
        return match.group(1).strip() if match else "System"
    
    def _load_document(self, file_path: str) -> bool:
        """
        Load and process the document.
        
        Args:
            file_path: Path to the requirements file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                self.add_error(f"File '{file_path}' not found.")
                return False
            
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Store raw text
            self.code_system.raw_text = text
            
            # Create chunker with settings
            chunker = TextChunker(
                chunk_size=self.settings.get("document", "chunk_size"),
                chunk_overlap=self.settings.get("document", "chunk_overlap")
            )
            
            # Create documents from text
            self.documents = chunker.create_documents(text, {"source": file_path})
            
            # Limit number of chunks if needed
            max_chunks = self.settings.get("document", "max_chunk_count")
            if len(self.documents) > max_chunks:
                self.add_warning(f"Document contains {len(self.documents)} chunks, limiting to {max_chunks}")
                self.documents = self.documents[:max_chunks]
            
            # Create vector store
            self.vector_embeddings.create_vector_store(self.documents)
            
            return True
            
        except Exception as e:
            self.add_error(f"Failed to load document: {str(e)}")
            return False
    
    def _analyze_domain(self) -> bool:
        """
        Analyze domain description to extract core concepts.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            prompt = f"""Analyze this domain in depth:
{self.domain_description}
Extract the following information to create a precise domain model:
1. Core Domain Entities: The primary domain objects (5-15, singular nouns, domain-specific)
2. Key Domain Relationships: Relationships between core entities with precise types
3. Key Domain Attributes: For each core entity, typical attributes with types and visibility
4. Key Domain Operations: For each core entity, main operations with signatures and visibility
5. Main Actors: Primary actors interacting with the system
6. Enumerations: Any enumerations with their values
Return as structured JSON with keys: "core_entities", "relationships", "attributes", "operations", "actors", "enumerations".
For "relationships", use format: [{{ "source": "EntityA", "target": "EntityB", "relationship": "verb phrase", "multiplicity": {{ "source": "1|0..1|*", "target": "1|0..1|*" }} }}]
For "attributes", use: {{ "Entity": [{{ "name": "attrName", "type": "String", "visibility": "+|-|#" }}] }}
For "operations", use: {{ "Entity": [{{ "name": "opName", "signature": "(param: Type): ReturnType", "visibility": "+|-|#" }}] }}
For "enumerations", use: {{ "EnumName": ["VALUE1", "VALUE2"] }}
Ensure JSON is valid and focuses on business domain concepts."""

            # Try to get relevant context to enrich the prompt
            try:
                docs = self.vector_embeddings.similarity_search(self.domain_description)
                domain_context = "\n\n".join([doc.page_content for doc in docs[:3]])
                full_prompt = f"{prompt}\n\nAdditional domain information:\n{domain_context}"
            except:
                full_prompt = prompt
            
            # Call LLM to analyze domain
            for attempt in range(3):
                try:
                    # Extract JSON result
                    result = self.llm_client.call_with_json_extraction(full_prompt)
                    
                    if result:
                        # Store core entities
                        self.code_system.core_domain_entities = result.get("core_entities", [])
                        
                        # Process relationships
                        relationships = result.get("relationships", [])
                        for rel in relationships:
                            source = rel.get("source", "")
                            target = rel.get("target", "")
                            relationship = rel.get("relationship", "")
                            multiplicity = rel.get("multiplicity", {"source": "1", "target": "*"})
                            
                            if source and target and relationship:
                                if source not in self.code_system.domain_relationships:
                                    self.code_system.domain_relationships[source] = []
                                    
                                self.code_system.domain_relationships[source].append(
                                    f"{relationship} {target} ({multiplicity['source']}:{multiplicity['target']})"
                                )
                        
                        # Store domain attributes, operations, actors, and enumerations
                        self.code_system.domain_attributes = result.get("attributes", {})
                        self.code_system.domain_operations = result.get("operations", {})
                        self.code_system.domain_actors = result.get("actors", [])
                        self.code_system.enumerations = result.get("enumerations", {})
                        
                        return True
                        
                except Exception as e:
                    self.add_warning(f"Domain analysis attempt {attempt+1} failed: {str(e)}")
            
            self.add_error("Failed to analyze domain after multiple attempts")
            return False
            
        except Exception as e:
            self.add_error(f"Domain analysis failed: {str(e)}")
            return False
    
    def execute(
        self, 
        file_path: str = None, 
        domain_description: str = None,
        output_dir: str = "output"
    ) -> PipelineResult:
        """
        Execute the domain modeling pipeline.
        
        Args:
            file_path: Path to the requirements file (if not already set up)
            domain_description: Description of the domain (if not already set up)
            output_dir: Directory to save outputs
            
        Returns:
            Pipeline execution result
        """
        self._start_execution()
        
        # Setup if not already done
        if file_path and domain_description and not self.documents:
            setup_success = self.setup(file_path, domain_description)
            if not setup_success:
                return self.create_result(False)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Prepare shared context for phases
            context = PhaseContext(
                code_system=self.code_system,
                vector_embeddings=self.vector_embeddings,
                documents=self.documents,
                api_key=self.api_key,
                model_name=self.model_name,
                domain_name=self.domain_name,
                metadata={
                    "banned_words": set(self.settings.get("model", "banned_words"))
                }
            )
            
            # Phase 1: Open Coding (Entity Extraction)
            self.add_message("\n1. Starting Open Coding phase - Extracting entities...")
            open_coding = OpenCodingPhase()
            open_result = open_coding.execute(context)
            
            # Add phase result
            self.add_metric("entities_extracted", open_result.metrics.get("final_entity_count", 0))
            
            # Phase 2: Axial Coding (Relationship Identification)
            self.add_message("\n2. Starting Axial Coding phase - Identifying relationships...")
            axial_coding = AxialCodingPhase()
            axial_result = axial_coding.execute(context)
            
            # Add phase result
            self.add_metric("relationships_identified", axial_result.metrics.get("relationship_count", 0))
            
            # Phase 3: Selective Coding (Model Refinement)
            self.add_message("\n3. Starting Selective Coding phase - Refining model...")
            selective_coding = SelectiveCodingPhase()
            selective_result = selective_coding.execute(context)
            
            # Add phase result
            self.add_metric("final_entity_count", selective_result.metrics.get("final_class_count", 0))
            self.add_metric("final_relationship_count", selective_result.metrics.get("final_relationship_count", 0))
            
            # Validate model
            self.add_message("\n4. Validating model...")
            validator = ModelValidator(self.code_system)
            validation_success = validator.validate(auto_fix=True)
            
            if not validation_success:
                self.add_warning("Model validation found issues that could not be automatically fixed")
            
            # Generate outputs
            self.add_message("\n5. Generating UML diagram and report...")
            plantuml_code, report = self._generate_outputs()
            
            # Save outputs
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save PlantUML diagram
            plantuml_file = os.path.join(
                output_dir, 
                f"{self.domain_name.lower()}_model_{timestamp}.puml"
            )
            with open(plantuml_file, "w", encoding="utf-8") as f:
                f.write(plantuml_code)
                
            self.add_message(f"PlantUML diagram saved to {plantuml_file}")
            
            # Save report
            report_file = os.path.join(
                output_dir, 
                f"{self.domain_name.lower()}_report_{timestamp}.md"
            )
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
                
            self.add_message(f"Traceability report saved to {report_file}")
            
            # Create final result
            entity_count = len(self.code_system.codes)
            relationship_count = len(self.code_system.relationships)
            
            self.add_message(
                f"\nPipeline completed. UML model contains {entity_count} "
                f"classes and {relationship_count} relationships"
            )
            
            # Record execution time
            execution_time = self._end_execution()
            
            # Create pipeline result
            result = PipelineResult(
                success=True,
                execution_time=execution_time,
                outputs={
                    "plantuml_code": plantuml_code,
                    "report": report,
                    "plantuml_file": plantuml_file,
                    "report_file": report_file
                },
                metrics=self.metrics,
                messages=self.messages,
                phase_results={
                    "open_coding": open_result,
                    "axial_coding": axial_result,
                    "selective_coding": selective_result
                }
            )
            
            return result
            
        except Exception as e:
            self.add_error(f"Pipeline execution failed: {str(e)}")
            execution_time = self._end_execution()
            
            # Create error result
            return PipelineResult(
                success=False,
                execution_time=execution_time,
                outputs={},
                metrics=self.metrics,
                messages=self.messages
            )
    
    def _generate_outputs(self) -> Tuple[str, str]:
        """
        Generate UML diagram and report.
        
        Returns:
            Tuple containing (plantuml_code, report_markdown)
        """
        # Generate PlantUML diagram
        style = self.settings.get("output", "diagram_style")
        plantuml_generator = PlantUMLGenerator(self.code_system, self.domain_name)
        plantuml_code = plantuml_generator.generate(style=style)
        
        # Generate traceability report
        include_metrics = self.settings.get("output", "include_metrics")
        report_generator = ReportGenerator(self.code_system, self.domain_name)
        report = report_generator.generate_markdown(include_metrics=include_metrics)
        
        return plantuml_code, report
    
    def get_code_system(self) -> CodeSystem:
        """
        Get the current code system.
        
        Returns:
            The code system
        """
        return self.code_system