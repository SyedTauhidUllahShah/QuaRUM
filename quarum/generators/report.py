"""
Traceability report generation module.

This module provides the ReportGenerator class for creating detailed
traceability reports that link domain model elements to their sources.
"""

import time
from typing import Dict, List, Any, Optional, Callable, TextIO

from quarum.core.code_system import CodeSystem
from quarum.core.enums import CSLRelationshipType


class ReportGenerator:
    """
    Generator for traceability reports from domain models.
    
    This class creates reports documenting the domain model elements
    and their traceability to the original requirements.
    """
    
    def __init__(self, code_system: CodeSystem, domain_name: str):
        """
        Initialize the report generator.
        
        Args:
            code_system: The domain model code system
            domain_name: Name of the domain
        """
        self.code_system = code_system
        self.domain_name = domain_name
        
    def generate_markdown(self, include_metrics: bool = True) -> str:
        """
        Generate a Markdown report for the domain model.
        
        Args:
            include_metrics: Whether to include model metrics
            
        Returns:
            Markdown report as string
        """
        report = [f"# Traceability Report for {self.domain_name} Domain Model\n"]
        report.append(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add domain description
        report.append("## Domain Description")
        report.append(self.code_system.domain_description)
        report.append("\n")
        
        # Add core entities
        report.append("## Core Entities")
        report.append(", ".join(self.code_system.core_domain_entities))
        report.append("\n")
        
        # Add metrics if requested
        if include_metrics:
            report.append("## Model Metrics")
            metrics = self._calculate_metrics()
            report.append(f"- **Total entities:** {metrics['total_entities']}")
            report.append(f"- **Classes:** {metrics['classes']}")
            report.append(f"- **Interfaces:** {metrics['interfaces']}")
            report.append(f"- **Abstract classes:** {metrics['abstract_classes']}")
            report.append(f"- **Enumerations:** {metrics['enumerations']}")
            report.append(f"- **Actors:** {metrics['actors']}")
            report.append(f"- **Relationships:** {metrics['relationships']}")
            report.append(f"- **Inheritance relationships:** {metrics['inheritance']}")
            report.append(f"- **Implementation relationships:** {metrics['implementation']}")
            report.append(f"- **Composition relationships:** {metrics['composition']}")
            report.append(f"- **Association relationships:** {metrics['association']}")
            report.append("\n")
        
        # Add classes and elements
        report.append("## Classes and Elements")
        for code in sorted(self.code_system.codes.values(), key=lambda x: x.name):
            # Skip banned terms
            if self._is_banned_term(code.name):
                continue
                
            report.append(f"### {code.name}")
            report.append(f"- **Type:** {', '.join(code.stereotypes)}")
            report.append(f"- **Definition:** {code.definition}")
            
            if code.attributes:
                report.append("- **Attributes:**")
                for attr in code.attributes:
                    report.append(f"  - {attr['visibility']} {attr['name']}: {attr['type']}")
                    
            if code.methods:
                report.append("- **Methods:**")
                for method in code.methods:
                    report.append(f"  - {method['visibility']} {method['name']}{method['signature']}")
                    
            if code.enum_values:
                report.append("- **Enum Values:**")
                for value in code.enum_values:
                    report.append(f"  - {value}")
            
            report.append("- **Evidence Locations:**")
            for loc in code.evidence_locations:
                report.append(f"  - {loc}")
                
            report.append("- **Confidence:** {:.2f}".format(code.confidence))
            
            if code.is_recommendation:
                report.append("- **Note:** Added based on domain analysis")
                
            if code.notes:
                report.append("- **Notes:**")
                for note in code.notes:
                    report.append(f"  - {note}")
                    
            report.append("\n")
        
        # Add relationships
        report.append("## Relationships")
        for rel in sorted(self.code_system.relationships, key=lambda r: r.id):
            source = self.code_system.codes.get(rel.source_code_id)
            target = self.code_system.codes.get(rel.target_code_id)
            
            # Skip if either entity doesn't exist or is banned
            if (not source or not target or 
                self._is_banned_term(source.name) or 
                self._is_banned_term(target.name)):
                continue
                
            report.append(f"### {source.name} -> {target.name}")
            report.append(f"- **Type:** {rel.relationship_type.value}")
            report.append(f"- **Name:** {rel.association_name}")
            report.append(f"- **Multiplicity:** {rel.multiplicity['source']} : {rel.multiplicity['target']}")
            
            report.append("- **Evidence:**")
            for chunk in rel.evidence_chunks:
                report.append(f"  - {chunk}")
                
            for loc in rel.evidence_locations:
                report.append(f"  - Location: {loc}")
                
            report.append("- **Confidence:** {:.2f}".format(rel.confidence))
            report.append("\n")
            
        return "\n".join(report)
    
    def generate_html(self, include_metrics: bool = True) -> str:
        """
        Generate an HTML report for the domain model.
        
        Args:
            include_metrics: Whether to include model metrics
            
        Returns:
            HTML report as string
        """
        # Convert markdown to HTML
        markdown = self.generate_markdown(include_metrics)
        
        # Add basic styling
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Traceability Report - {self.domain_name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; margin: 2em; max-width: 1000px; margin: 0 auto; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }",
            "h3 { color: #2980b9; }",
            ".confidence-high { color: green; }",
            ".confidence-medium { color: orange; }",
            ".confidence-low { color: red; }",
            "pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }",
            "code { font-family: monospace; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        # Convert markdown headings, lists, and paragraphs to HTML
        lines = markdown.split("\n")
        in_list = False
        
        for line in lines:
            if line.startswith("# "):
                html.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                if not in_list:
                    html.append("<ul>")
                    in_list = True
                html.append(f"<li>{line[2:]}</li>")
            elif line.startswith("  - "):
                html.append(f"<li style='margin-left: 20px;'>{line[4:]}</li>")
            elif line.strip() == "" and in_list:
                html.append("</ul>")
                html.append("<p></p>")
                in_list = False
            elif line.strip() == "":
                html.append("<p></p>")
            else:
                html.append(f"<p>{line}</p>")
                
        if in_list:
            html.append("</ul>")
            
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def save_report(
        self, 
        output_path: str, 
        format: str = "markdown", 
        include_metrics: bool = True
    ) -> None:
        """
        Save the report to a file.
        
        Args:
            output_path: Path to save the report
            format: Format ('markdown' or 'html')
            include_metrics: Whether to include metrics
        """
        if format.lower() == "html":
            content = self.generate_html(include_metrics)
        else:
            content = self.generate_markdown(include_metrics)
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _calculate_metrics(self) -> Dict[str, int]:
        """
        Calculate metrics about the model.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "total_entities": 0,
            "classes": 0,
            "interfaces": 0,
            "abstract_classes": 0,
            "enumerations": 0,
            "actors": 0,
            "relationships": len(self.code_system.relationships),
            "inheritance": 0,
            "implementation": 0,
            "composition": 0,
            "association": 0,
        }
        
        # Count entity types
        for code in self.code_system.codes.values():
            if self._is_banned_term(code.name):
                continue
                
            metrics["total_entities"] += 1
            
            if "actor" in code.stereotypes:
                metrics["actors"] += 1
            elif code.is_enumeration:
                metrics["enumerations"] += 1
            elif code.is_interface:
                metrics["interfaces"] += 1
            elif code.is_abstract:
                metrics["abstract_classes"] += 1
            else:
                metrics["classes"] += 1
        
        # Count relationship types
        for rel in self.code_system.relationships:
            if rel.relationship_type == CSLRelationshipType.IS_A:
                metrics["inheritance"] += 1
            elif rel.relationship_type == CSLRelationshipType.IMPLEMENTATION:
                metrics["implementation"] += 1
            elif rel.relationship_type == CSLRelationshipType.IS_PART_OF:
                metrics["composition"] += 1
            else:
                metrics["association"] += 1
                
        return metrics
    
    def _is_banned_term(self, name: str) -> bool:
        """
        Check if a name is a banned generic term.
        
        Args:
            name: Name to check
            
        Returns:
            True if name is banned, False otherwise
        """
        banned_words = set(["system", "component", "generic", "entity", "item", "object", "module"])
        nl = name.strip().lower()
        
        # Check if in banned words set
        if nl in banned_words:
            return True
        
        # Check if too short
        if len(nl) <= 2:
            return True
            
        return False
    
    def generate_metrics_report(self) -> str:
        """
        Generate a standalone metrics report.
        
        Returns:
            Metrics report as string
        """
        metrics = self._calculate_metrics()
        
        report = [f"# Domain Model Metrics for {self.domain_name}\n"]
        report.append(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Entity Metrics")
        report.append(f"- Total entities: {metrics['total_entities']}")
        report.append(f"- Regular classes: {metrics['classes']}")
        report.append(f"- Interfaces: {metrics['interfaces']}")
        report.append(f"- Abstract classes: {metrics['abstract_classes']}")
        report.append(f"- Enumerations: {metrics['enumerations']}")
        report.append(f"- Actors: {metrics['actors']}")
        report.append("\n")
        
        report.append("## Relationship Metrics")
        report.append(f"- Total relationships: {metrics['relationships']}")
        report.append(f"- Inheritance (is-a): {metrics['inheritance']}")
        report.append(f"- Implementation: {metrics['implementation']}")
        report.append(f"- Composition (part-of): {metrics['composition']}")
        report.append(f"- Association and others: {metrics['association']}")
        
        # Calculate derived metrics
        inheritance_percent = (metrics['inheritance'] / metrics['relationships'] * 100) if metrics['relationships'] > 0 else 0
        report.append(f"- Inheritance percentage: {inheritance_percent:.1f}%")
        
        relationships_per_entity = metrics['relationships'] / metrics['total_entities'] if metrics['total_entities'] > 0 else 0
        report.append(f"- Relationships per entity: {relationships_per_entity:.2f}")
        
        return "\n".join(report)