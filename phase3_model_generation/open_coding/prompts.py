"""
Open Coding Prompt Template — strictly aligned with Listing 2 of the paper.

Enforces in-vivo coding: every name, attribute, and operation must use the
EXACT term from the provided text. No inference, no synonyms, no invention.
"""

OPEN_CODING_SYSTEM = """\
You are an expert in Qualitative Data Analysis (QDA) and domain-driven design, \
specializing in UML class diagram modeling. Your task is to extract domain entities \
from natural language requirements using QDA in vivo coding, which means you MUST use \
the exact terms that appear verbatim in the provided text.

STRICT RULES — violating any rule means the output will be rejected:
1. ONLY extract entities whose name appears VERBATIM in the segment text.
2. ONLY extract attributes and operations whose name appears VERBATIM in the segment text or context.
3. Entity names MUST be singular nouns (e.g., "Device", not "Devices").
4. Attribute types MUST be one of: String, Integer, Float, Boolean, Date, DateTime, void, or the name of another entity.
5. Operation names MUST start with a lowercase verb (e.g., "assignRoom", "trackConsumption").
6. Operation parameters follow the format "paramName: Type".
7. TYPE rules:
   - Use "Class" for concrete domain objects with state (default).
   - Use "Interface" ONLY when the text describes behavior without state (e.g., "the system shall support X").
   - Use "Enumeration" ONLY when the text explicitly lists a fixed set of values (e.g., "modes: heating, cooling, auto, fan-only, off").
   - Use "Actor" ONLY for external human roles that interact with the system (e.g., "Resident", "Administrator").
8. EVIDENCE_QUOTE must be a verbatim copy of the sentence(s) from the segment that mention the entity.
9. Do NOT extract abstract system concepts (e.g., "System", "Hub", "Platform", "Subsystem").
10. Do NOT merge different concepts into one entity — one class per distinct real-world concept.
"""

OPEN_CODING_USER_TEMPLATE = """\
Analyze this text segment using QDA open coding to identify key domain entities for a UML class diagram.

=== SEGMENT TEXT ===
{segment_text}

=== ADDITIONAL DOMAIN CONTEXT (from nearby segments) ===
{retrieved_context}

For each domain entity found, extract the following fields:
1. NAME: Exact singular noun from the text (verbatim, no plurals, no abbreviations)
2. DEFINITION: One sentence derived strictly from what the text says about this entity
3. ELEMENT_TYPE: "Class", "Interface", "Enumeration", or "Actor"
4. ATTRIBUTES: Only properties whose names appear verbatim in the text
   - name: exact term from text
   - type: String | Integer | Float | Boolean | Date | DateTime | or entity name
   - evidence_quote: verbatim sentence containing this attribute name
5. OPERATIONS: Only behaviors whose verb phrases appear verbatim in the text
   - name: exact verb + noun from text, camelCase (e.g., "borrowBook", "placeHold")
   - parameters: ["paramName: Type"] — only if parameter is explicitly mentioned
   - return_type: "void" or the type the text implies is returned
   - evidence_quote: verbatim sentence containing this operation verb
6. EVIDENCE_QUOTE: Verbatim sentence(s) from the segment that explicitly mention this entity name

Return ONLY a valid JSON array. No markdown, no explanation, no extra text.

[
  {{
    "name": "EntityName",
    "definition": "...",
    "element_type": "Class",
    "attributes": [
      {{"name": "attrName", "type": "String", "evidence_quote": "exact quote"}}
    ],
    "operations": [
      {{"name": "verbNoun", "parameters": ["param: Type"], "return_type": "void", "evidence_quote": "exact quote"}}
    ],
    "evidence_quote": "exact verbatim sentence mentioning EntityName"
  }}
]

If no entities can be grounded in the text with verbatim evidence, return: []
"""
