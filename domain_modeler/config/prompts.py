"""
Prompt templates for domain modeling.

This module provides standardized prompt templates for different
tasks in the domain modeling process, ensuring consistency and
supporting customization.
"""

import os
from typing import Dict, Any, Optional

class PromptTemplates:
    """
    Manages prompt templates for domain modeling.
    
    This class provides centralized storage and access to prompt
    templates used in the domain modeling process, with support
    for customization and overrides.
    """
    
    # System prompt
    DEFAULT_SYSTEM_PROMPT = """You are a qualitative data analysis-based expert in UML and domain-driven design.
Your task is to extract domain models from natural language requirements.
Focus on creating clean, accurate, and useful class diagrams that properly represent the domain.
Identify the most important classes, attributes, operations, and relationships.
Ensure all JSON outputs are valid, with proper syntax (e.g., commas, brackets).
Use UML conventions for attributes and methods with visibility (+, -, #) and proper types."""

    # Task-specific templates
    DEFAULT_TEMPLATES = {
        "domain_analysis": """Analyze this domain in depth:
{domain_description}
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
Ensure JSON is valid and focuses on business domain concepts.""",

        "entity_extraction": """
# Domain Entity Extraction for {domain_name}
Analyze this text segment and extract key domain entities for a UML class diagram:
{text_chunk}
## Extraction Guidelines:
1. Focus on entities and concepts from the text
2. For each entity identify:
   - NAME: The name as mentioned in the text
   - DEFINITION: Description based on the text
   - ATTRIBUTES: Properties mentioned or implied in the text
   - OPERATIONS: Methods/behaviors mentioned or implied
   - TYPE: Class, Interface, or Enumeration
   - EVIDENCE: Text that mentions this entity
   - EXTRACTED_TEXT: Complete sentence where entity appears
   - CONFIDENCE: Rate evidence strength (0.5-1.0)
3. Include actors and interfaces
4. Look for attributes and methods that make sense in context
{domain_context}
Return JSON:
{{
"entities": [
    {{
    "name": "EntityName",
    "definition": "Description from text",
    "type": "Class|Interface|Enumeration|Actor",
    "attributes": [{{ "name": "attrName", "type": "Type", "visibility": "+|-|#" }}],
    "operations": [{{ "name": "opName", "signature": "(param: Type): ReturnType", "visibility": "+|-|#" }}],
    "evidence": "Text quote that mentions this entity",
    "extracted_text": "Complete sentence where entity appears",
    "confidence": 0.8
    }}
]
}}
For enumerations, include "enumValues": ["VALUE1", "VALUE2"] if listed in text.""",

        "relationship_analysis": """
# UML Relationship Analysis
## Class Pairs:
{class_pairs_json}
{domain_rel_hints}
## Analysis Task:
Determine UML relationships between each pair based on textual evidence or domain knowledge.
Look for explicit or implicit relationships in the context.
Consider domain-specific patterns and typical relationships.
Valid relationship types:
- IS_A: Child inherits from parent
- IS_PART_OF: Composition relationship
- IMPLEMENTS: Class implements interface
- DEPENDS_ON: One class requires another
- USES: One class uses another
- ASSOCIATION: General relationship
- MANAGES: One class manages another
- CREATES: One class creates another
- NONE: No relationship found
Guidelines:
- Find relationships based on text evidence when available
- Where text is unclear, apply domain knowledge from the context
- Confidence should reflect evidence strength (0.5-1.0)
- For domain-driven relationships without direct evidence, use confidence 0.6
- If no relationship exists, use NONE
Output JSON:
{{
"relationships": [
    {{
    "sourceId": "source_id",
    "targetId": "target_id",
    "type": "IS_A|IS_PART_OF|IMPLEMENTS|DEPENDS_ON|USES|ASSOCIATION|MANAGES|CREATES|NONE",
    "association_name": "verb phrase (e.g., 'creates', 'manages')",
    "evidence": "Text supporting this relationship",
    "confidence": 0.8,
    "multiplicity": {{ "source": "1|0..1|*", "target": "1|0..1|*" }}
    }}
]
}}""",

        "interface_identification": """
# Interface and Abstract Class Identification
## Current Classes:
{classes_json}
## Identification Task:
Identify which classes should be interfaces or abstract classes based ONLY on direct evidence in the extracted_text field.
DO NOT infer or guess - only mark as interface/abstract if clearly indicated.
## Output Format:
{{
"interfaces": [
    {{
    "id": "class_id",
    "evidence": "Exact text evidence supporting interface classification"
    }}
],
"abstractClasses": [
    {{
    "id": "class_id",
    "evidence": "Exact text evidence supporting abstract classification"
    }}
]
}}""",

        "class_enrichment": """
# Enrich UML Class Features for {domain_name}
## Classes to Enrich:
{class_info_json}
## Enrichment Task:
Provide domain-specific attributes and operations for each class.
Guidelines:
- Attributes: Include name, type (String, Integer, etc.), visibility (+, -, #)
- Operations: Include name, signature with parameters/return type, visibility
- Interfaces: Only methods, no attributes
- Ensure domain relevance
- Use camelCase for methods, lowerCamelCase for attributes
Output JSON:
{{
"enrichedClasses": [
    {{
    "id": "class_id",
    "attributes": [{{"name": "attrName", "type": "Type", "visibility": "+|-|#"}}],
    "operations": [{{"name": "opName", "signature": "(param: Type): ReturnType", "visibility": "+|-|#"}}]
    }}
]
}}"""
    }
    
    def __init__(self, custom_templates_path: Optional[str] = None):
        """
        Initialize prompt templates.
        
        Args:
            custom_templates_path: Optional path to a directory with custom templates
        """
        # Start with default templates
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.templates = self.DEFAULT_TEMPLATES.copy()
        
        # Load custom templates if provided
        if custom_templates_path and os.path.isdir(custom_templates_path):
            self._load_custom_templates(custom_templates_path)
    
    def _load_custom_templates(self, templates_path: str) -> None:
        """
        Load custom templates from files.
        
        Args:
            templates_path: Path to directory with template files
        """
        # Look for system prompt
        system_prompt_path = os.path.join(templates_path, "system_prompt.txt")
        if os.path.isfile(system_prompt_path):
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        
        # Look for task-specific templates
        for task in self.templates.keys():
            template_path = os.path.join(templates_path, f"{task}.txt")
            if os.path.isfile(template_path):
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.templates[task] = f.read()
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt.
        
        Returns:
            System prompt text
        """
        return self.system_prompt
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt.
        
        Args:
            prompt: New system prompt text
        """
        self.system_prompt = prompt
    
    def get_template(self, task: str) -> str:
        """
        Get a prompt template by task name.
        
        Args:
            task: Task name
            
        Returns:
            Template text
            
        Raises:
            ValueError: If task is not found
        """
        if task not in self.templates:
            raise ValueError(f"Unknown task template: {task}")
            
        return self.templates[task]
    
    def set_template(self, task: str, template: str) -> None:
        """
        Set a prompt template.
        
        Args:
            task: Task name
            template: Template text
        """
        self.templates[task] = template
    
    def add_template(self, task: str, template: str) -> None:
        """
        Add a new prompt template.
        
        Args:
            task: Task name
            template: Template text
        """
        self.templates[task] = template
    
    def format_template(self, task: str, **kwargs: Any) -> str:
        """
        Format a template with parameters.
        
        Args:
            task: Task name
            **kwargs: Parameters to inject
            
        Returns:
            Formatted template
            
        Raises:
            ValueError: If task is not found
        """
        template = self.get_template(task)
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Missing parameter for template '{task}': {missing_key}")
    
    def get_all_task_names(self) -> list:
        """
        Get all available task names.
        
        Returns:
            List of task names
        """
        return list(self.templates.keys())
    
    def save_to_directory(self, directory_path: str) -> bool:
        """
        Save all templates to a directory.
        
        Args:
            directory_path: Path to save templates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory_path, exist_ok=True)
            
            # Save system prompt
            with open(os.path.join(directory_path, "system_prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(self.system_prompt)
            
            # Save task-specific templates
            for task, template in self.templates.items():
                with open(os.path.join(directory_path, f"{task}.txt"), 'w', encoding='utf-8') as f:
                    f.write(template)
                    
            return True
            
        except Exception as e:
            print(f"Error saving templates: {str(e)}")
            return False
    
    def reset(self) -> None:
        """Reset all templates to defaults."""
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.templates = self.DEFAULT_TEMPLATES.copy()
    
    def reset_template(self, task: str) -> bool:
        """
        Reset a specific template to default.
        
        Args:
            task: Task name
            
        Returns:
            True if successful, False if task not found
        """
        if task not in self.DEFAULT_TEMPLATES:
            return False
            
        self.templates[task] = self.DEFAULT_TEMPLATES[task]
        return True