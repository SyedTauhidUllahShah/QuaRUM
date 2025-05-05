"""
Code element representation module.

This module defines the Code class, which represents classes,
interfaces, enumerations, and other code elements in the domain model.
"""

from quarum.core.relationship import CodeRelationship


class Code:
    """
    Represents a code element in the domain model.

    A code element can be a class, interface, enumeration, or any other
    structural element that appears in a UML class diagram. It contains
    attributes, methods, relationships, and metadata about how it was
    identified in the requirements.
    """

    def __init__(
        self,
        code_id: str,
        name: str,
        definition: str = "",
        is_recommendation: bool = False,
    ):
        """
        Initialize a new Code element.

        Args:
            code_id: Unique identifier for this code element
            name: Name of the code element (class/interface/enum name)
            definition: Description or definition of the element
            is_recommendation: Whether this element was suggested rather than
                              directly extracted from requirements
        """
        self.id = code_id
        self.name = name
        self.definition = definition
        self.attributes: list[dict[str, str]] = []
        self.methods: list[dict[str, str]] = []
        self.stereotypes: list[str] = []
        self.trace_sources: list[str] = []
        self.evidence_chunks: list[str] = []
        self.is_interface: bool = False
        self.is_abstract: bool = False
        self.is_enumeration: bool = False
        self.enum_values: list[str] = []
        self.confidence: float = 0.0
        self.evidence_locations: list[str] = []
        self.outgoing_relationships: list["CodeRelationship"] = []
        self.incoming_relationships: list["CodeRelationship"] = []
        self.relevance_score: float = 0.0
        self.extracted_text: str = ""
        self.is_recommendation = is_recommendation
        self.notes: list[str] = []

    def add_attribute(self, name: str, attr_type: str, visibility: str = "+") -> None:
        """
        Add an attribute to this code element.

        Args:
            name: Name of the attribute
            attr_type: Data type of the attribute
            visibility: Visibility modifier (+ public, - private, # protected)
        """
        self.attributes.append(
            {"name": name, "type": attr_type, "visibility": visibility}
        )

    def add_method(self, name: str, signature: str, visibility: str = "+") -> None:
        """
        Add a method to this code element.

        Args:
            name: Name of the method
            signature: Method signature including parameters and return type
            visibility: Visibility modifier (+ public, - private, # protected)
        """
        self.methods.append(
            {"name": name, "signature": signature, "visibility": visibility}
        )

    def add_evidence(self, chunk: str, location: str) -> None:
        """
        Add evidence of this element from the requirements.

        Args:
            chunk: Text excerpt supporting this element
            location: Source location reference (e.g., file:line)
        """
        self.evidence_chunks.append(chunk)
        self.evidence_locations.append(location)

    def set_as_interface(self) -> None:
        """Mark this code element as an interface."""
        self.is_interface = True
        if "interface" not in self.stereotypes:
            self.stereotypes.append("interface")

    def set_as_abstract(self) -> None:
        """Mark this code element as an abstract class."""
        self.is_abstract = True
        if "abstract" not in self.stereotypes:
            self.stereotypes.append("abstract")

    def set_as_enumeration(self) -> None:
        """Mark this code element as an enumeration."""
        self.is_enumeration = True
        if "enumeration" not in self.stereotypes:
            self.stereotypes.append("enumeration")

    def add_enum_value(self, value: str) -> None:
        """
        Add an enumeration value to this element.

        Args:
            value: Enum value to add
        """
        if value not in self.enum_values:
            self.enum_values.append(value)
