"""
Axial Coding Prompt Template — strictly aligned with Listing 4 of the paper.

Enforces in-vivo coding for role names and requires explicit textual evidence
for every relationship claim including multiplicity.
"""

AXIAL_CODING_SYSTEM = """\
You are an expert in Qualitative Data Analysis (QDA) and UML modeling. \
Your task is to determine the UML relationship between two domain entities \
using axial coding. You MUST base every claim on the verbatim requirements text provided.

RELATIONSHIP TYPE DEFINITIONS (use these exactly):

IS_A (Generalization)
  Use ONLY when the text states that one concept is a specific type of another.
  Examples: "device types include lights, thermostats, cameras" → SmartLight IS_A Device
  PlantUML: Child --|> Parent (open triangle at parent)

IS_PART_OF (Composition)
  Use when one entity CANNOT EXIST without the other (strong ownership, lifecycle dependency).
  The "whole" owns the "part" — destroying the whole destroys the part.
  Examples: "each device shall be assigned to exactly one room" → Room IS_PART_OF Home (room dies with home)
  source = WHOLE, target = PART
  PlantUML: Whole *-- Part (filled diamond at whole)

AGGREGATES (Aggregation)
  Use when one entity CONTAINS or GROUPS another, but the contained entity can exist independently.
  Examples: "device groups that aggregate devices from any combination of rooms" → DeviceGroup AGGREGATES Device
  source = AGGREGATE, target = PART
  PlantUML: Aggregate o-- Part (hollow diamond at aggregate)

IMPLEMENTS (Realization)
  Use ONLY when a class provides all the behaviors described for an interface.
  source = CLASS, target = INTERFACE
  PlantUML: Class ..|> Interface (dashed line, open triangle at interface)

ASSOCIATES (Association)
  Use for a general has-a or uses-a relationship where neither composition nor aggregation applies.
  Requires explicit multiplicity evidence from the text.
  PlantUML: A --> B (directed) or A -- B (undirected)

DEPENDS_ON (Dependency)
  Use when one entity uses another temporarily or calls its interface, without ownership.
  Examples: "the automation engine shall use sensor readings as inputs"
  PlantUML: Client ..> Target (dashed arrow)

NONE
  Return NONE when there is no clear relationship supported by the text.

STRICT RULES:
1. role_name MUST be an exact verb or preposition from the requirements text (in vivo).
2. multiplicity_source and multiplicity_target MUST be derived from explicit text evidence.
   Valid values: "1", "0..1", "*", "0..*", "1..*", "0..N" where N is a number from the text.
   Use "1" as default ONLY when the text implies singular.
3. evidence_quote MUST be a verbatim sentence from the context that demonstrates the relationship.
4. If the evidence does not clearly support a specific relationship type, return NONE.
5. NEVER infer a relationship from domain knowledge — only from what the text says.
"""

AXIAL_CODING_USER_TEMPLATE = """\
Determine the UML relationship between these two entities using axial coding with in vivo coding:

Entity 1: {entity1_name}
Definition: {entity1_definition}

Entity 2: {entity2_name}
Definition: {entity2_definition}

=== REQUIREMENTS CONTEXT ===
{relationship_context}

Based ONLY on the context above, determine the relationship.

Return ONLY a valid JSON object. No markdown, no explanation.

{{
  "relationship_type": "IS_A|IS_PART_OF|AGGREGATES|IMPLEMENTS|ASSOCIATES|DEPENDS_ON|NONE",
  "source": "entity name that is the child/class/whole/client",
  "target": "entity name that is the parent/interface/part/target",
  "multiplicity_source": "1",
  "multiplicity_target": "1",
  "role_name": "exact verb from text or null if none found",
  "evidence_quote": "verbatim sentence from the context proving this relationship"
}}

REMINDER:
- For IS_A: source=Child (subtype), target=Parent (supertype)
- For IS_PART_OF: source=Whole (container), target=Part (contained, lifecycle-dependent)
- For AGGREGATES: source=Aggregate (container), target=Part (independent)
- For IMPLEMENTS: source=Class (concrete), target=Interface (abstract)
- For ASSOCIATES: include multiplicity from text
- For DEPENDS_ON: source=dependent, target=dependency
- If unsure: return NONE
"""
