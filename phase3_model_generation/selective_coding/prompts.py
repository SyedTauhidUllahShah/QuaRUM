"""
Selective Coding Prompt Templates — strictly aligned with Algorithm 3 of the paper.

Fix prompt: resolve item-scoped structural issues with verbatim evidence.
Enrich prompt: add missing but evidenced attributes/operations using in-vivo terms.
"""

SELECTIVE_FIX_SYSTEM = """\
You are an expert in Qualitative Data Analysis (QDA) and UML modeling. \
Your task is to resolve a specific structural issue in a UML domain model \
using targeted evidence from requirements text.

STRICT RULES:
1. You may ONLY propose a fix that is directly supported by the requirements context.
2. Use ONLY in-vivo terms (exact words from the text).
3. Propose the SMALLEST possible fix — do not change anything not directly implicated by the issue.
4. If the requirements context does not support any fix, return {{"action": "none"}}.
5. Structural violations (circular inheritance, type incompatibility) that cannot be resolved
   from the text must be resolved by removing the problematic element (action: "remove").
"""

SELECTIVE_FIX_USER_TEMPLATE = """\
A structural issue was detected in the UML model. Resolve ONLY this specific issue.

Issue type: {issue_type}
Affected element(s): {affected_elements}
Issue description: {issue_description}

=== REQUIREMENTS CONTEXT ===
{context_text}

=== CURRENT ELEMENT STATE ===
{element_state}

Return ONLY a valid JSON object. No markdown, no explanation.

{{
  "action": "update|remove|add|none",
  "element_type": "entity|relationship|attribute|operation",
  "element_name": "name of the element to modify",
  "changes": {{
    "field_name": "new_value"
  }},
  "evidence_quote": "verbatim sentence from context supporting this fix, or empty string"
}}

If no fix is supported by the evidence, return: {{"action": "none"}}
"""

SELECTIVE_ENRICH_SYSTEM = """\
You are an expert in Qualitative Data Analysis (QDA) and UML modeling. \
Your task is to enrich a UML entity with missing attributes or operations \
that are explicitly mentioned in the requirements context.

STRICT RULES:
1. Add ONLY attributes and operations whose names appear VERBATIM in the requirements context.
2. Use in-vivo naming: exact terms from the text, camelCase for operations.
3. Attribute types must be: String, Integer, Float, Boolean, Date, DateTime, or another entity name.
4. Operations must start with a lowercase verb.
5. evidence_quote must contain the exact text that mentions the new attribute or operation.
6. Do NOT add general-purpose CRUD operations unless the text explicitly describes them.
7. Return empty arrays if nothing new can be grounded in the evidence.
"""

SELECTIVE_ENRICH_USER_TEMPLATE = """\
The following UML entity needs to be enriched with missing details from the requirements.

Entity: {entity_name}
Current attributes: {current_attributes}
Current operations: {current_operations}

=== REQUIREMENTS CONTEXT ===
{context_text}

Based ONLY on the context above, identify attributes and operations that:
- Are explicitly mentioned in the text
- Are NOT already in the current lists above
- Have their names appear VERBATIM in the text

Return ONLY a valid JSON object. No markdown, no explanation.

{{
  "entity_name": "{entity_name}",
  "new_attributes": [
    {{"name": "exactName", "type": "String|Integer|Float|Boolean|Date|DateTime|EntityName", "evidence_quote": "verbatim sentence"}}
  ],
  "new_operations": [
    {{"name": "verbNoun", "parameters": ["paramName: Type"], "return_type": "void|Type", "evidence_quote": "verbatim sentence"}}
  ]
}}
"""
