"""
Prompt builder module for LLM interactions.

This module provides tools for constructing effective prompts
for language models used in the domain modeling process.
"""

from typing import Any, Optional


class PromptBuilder:
    """
    Builds and manages prompts for LLM interactions.

    This class provides methods to create structured prompts
    for various domain modeling tasks, with support for templates,
    context injection, and domain-specific customization.
    """

    def __init__(self, domain_name: str, system_prompt: Optional[str] = None):
        """
        Initialize the prompt builder.

        Args:
            domain_name: Name of the domain being modeled
            system_prompt: Optional base system prompt
        """
        self.domain_name = domain_name
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.templates = self._initialize_templates()

    def _default_system_prompt(self) -> str:
        """
        Provide the default system prompt for domain modeling.

        Returns:
            Default system prompt text
        """
        return """You are a qualitative data analysis-based expert in UML and domain-driven design.
Your task is to extract domain models from natural language requirements.
Focus on creating clean, accurate, and useful class diagrams that properly represent the domain.
Identify the most important classes, attributes, operations, and relationships.
Ensure all JSON outputs are valid, with proper syntax (e.g., commas, brackets).
Use UML conventions for attributes and methods with visibility (+, -, #) and proper types."""

    def _initialize_templates(self) -> dict[str, str]:
        """
        Initialize the standard prompt templates.

        Returns:
            dictionary of prompt templates by task
        """
        return {
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
}}""",
        }

    def build_prompt(self, task: str, **kwargs) -> str:
        """
        Build a prompt for a specific task with parameters.

        Args:
            task: The task template to use
            **kwargs: Parameters to inject into the template

        Returns:
            Formatted prompt text
        """
        if task not in self.templates:
            raise ValueError(f"Unknown task template: {task}")

        # Always include domain name
        kwargs["domain_name"] = self.domain_name

        # Format template with provided arguments
        template = self.templates[task]
        return template.format(**kwargs)

    def add_context_to_prompt(self, prompt: str, context: str) -> str:
        """
        Add additional context to a prompt.

        Args:
            prompt: The base prompt
            context: Context to append

        Returns:
            Prompt with added context
        """
        return f"{prompt}\n\nRelevant domain context:\n{context}"

    def add_template(self, name: str, template: str) -> None:
        """
        Add a new prompt template.

        Args:
            name: Name of the template
            template: Template text with {param} placeholders
        """
        self.templates[name] = template

    def escape_braces_for_format(self, text: str) -> str:
        """
        Escape braces for string formatting.

        Args:
            text: Text that may contain braces

        Returns:
            Text with braces escaped
        """
        return text.replace("{", "{{").replace("}", "}}")

    def add_examples_to_prompt(
        self, prompt: str, examples: list[dict[str, Any]]
    ) -> str:
        """
        Add few-shot examples to a prompt.

        Args:
            prompt: Base prompt
            examples: list of example dictionaries

        Returns:
            Prompt with examples
        """
        example_text = "\n\n## Examples:\n"

        for i, example in enumerate(examples):
            example_text += f"\nExample {i+1}:\n"
            example_text += f"Input: {example.get('input', '')}\n"
            example_text += f"Output: {example.get('output', '')}\n"

        return prompt + example_text
