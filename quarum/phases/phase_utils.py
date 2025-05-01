"""
Utility classes and functions for qualitative analysis phases.

This module provides common utilities used across the different coding
phases, including context management, result tracking, and helper functions.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import json
from langchain.docstore.document import Document

from quarum.core.code_system import CodeSystem
from quarum.nlp.embeddings import VectorEmbeddings


@dataclass
class PhaseContext:
    """
    Context information shared across analysis phases.
    
    This class encapsulates the shared state and resources needed by
    different analysis phases, promoting loose coupling between phases.
    """
    
    # The code system being built
    code_system: CodeSystem
    
    # Vector embeddings for semantic search
    vector_embeddings: VectorEmbeddings
    
    # Documents being analyzed
    documents: List[Document]
    
    # LLM API key
    api_key: str
    
    # LLM model name
    model_name: str
    
    # Domain name
    domain_name: str
    
    # Additional metadata/context
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize additional properties after construction."""
        if "banned_words" not in self.metadata:
            self.metadata["banned_words"] = set([
                "system", "component", "generic", "entity", 
                "item", "object", "module"
            ])


@dataclass
class PhaseResult:
    """
    Results and metrics from an analysis phase.
    
    This class captures the outcomes and performance metrics
    of each analysis phase for evaluation and refinement.
    """
    
    # Phase name
    phase_name: str
    
    # Success status
    success: bool
    
    # Metrics and statistics
    metrics: Dict[str, Any]
    
    # Messages and notes
    messages: List[str]
    
    # Artifacts produced
    artifacts: Dict[str, Any]
    
    # Execution time in seconds
    execution_time: float
    
    def add_message(self, message: str) -> None:
        """Add a message to the results."""
        self.messages.append(message)
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the results."""
        self.metrics[name] = value
    
    def add_artifact(self, name: str, artifact: Any) -> None:
        """Add an artifact to the results."""
        self.artifacts[name] = artifact


@runtime_checkable
class AnalysisPhase(Protocol):
    """Protocol defining the interface for analysis phases."""
    
    def execute(self, context: PhaseContext) -> PhaseResult:
        """
        Execute the analysis phase.
        
        Args:
            context: The shared context information
            
        Returns:
            Results of the phase execution
        """
        ...


class BasePhase(ABC):
    """
    Base class for implementing analysis phases.
    
    This abstract class provides common functionality for all
    analysis phases, including helper methods and error handling.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base phase.
        
        Args:
            name: Name of the phase
        """
        self.name = name
        self.messages = []
        self.metrics = {}
    
    @abstractmethod
    def execute(self, context: PhaseContext) -> PhaseResult:
        """
        Execute the analysis phase (to be implemented by subclasses).
        
        Args:
            context: The shared context information
            
        Returns:
            Results of the phase execution
        """
        pass
    
    def _extract_json_from_response(self, response: str) -> Optional[dict]:
        """
        Extract JSON object from LLM response.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed JSON object or None if extraction failed
        """
        try:
            # Clean the response to extract JSON
            cleaned_response = re.sub(r"```(?:json)?\n|```", "", response).strip()
            
            # Find JSON block using regex
            json_block = re.findall(r"\{[\s\S]*\}", cleaned_response)
            if not json_block:
                self.messages.append("No JSON object found in response")
                return None
            
            # Get the largest JSON block (likely the main content)
            json_str = max(json_block, key=len)
            
            # Fix common JSON syntax issues
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
            
            # Parse JSON
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            self.messages.append(f"Failed to parse JSON: {str(e)}")
            return None
    
    def _verify_text_presence(self, term: str, raw_text: str) -> bool:
        """
        Check if a term appears in the raw text.
        
        Args:
            term: Term to search for
            raw_text: Text to search in
            
        Returns:
            True if term is found, False otherwise
        """
        if not term:
            return False
        
        # Create a pattern that matches whole words, allowing for plural forms
        pattern = re.compile(r'\b' + re.escape(term) + r'(?:s)?\b', re.IGNORECASE)
        return bool(pattern.search(raw_text))
    
    def _is_banned_term(self, name: str, banned_words: set) -> bool:
        """
        Check if a name is a banned generic term.
        
        Args:
            name: Name to check
            banned_words: Set of banned words
            
        Returns:
            True if name is banned, False otherwise
        """
        nl = name.strip().lower()
        
        # Check if in banned words set
        if nl in banned_words:
            return True
        
        # Check if too short
        if len(nl) <= 2:
            return True
            
        return False
    
    def get_evidence_from_query(
        self, 
        query: str, 
        context: PhaseContext, 
        k: int = 3
    ) -> List[Document]:
        """
        Get semantically relevant documents for a query.
        
        Args:
            query: Query text
            context: Phase context
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            return context.vector_embeddings.similarity_search(query, k=k)
        except Exception as e:
            self.messages.append(f"Error retrieving documents: {str(e)}")
            return []