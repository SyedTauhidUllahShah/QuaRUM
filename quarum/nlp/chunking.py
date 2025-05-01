"""
Text chunking module for processing documents.

This module provides methods to split text documents into
manageable chunks for analysis while preserving context.
"""

from collections.abc import Callable
from typing import Any
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class TextChunker:
    """
    Handles chunking of text documents for processing.

    This class provides different strategies for splitting
    text into manageable chunks while maintaining appropriate
    context for NLP analysis.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = None,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
            separators: Optional list of separator strings for chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: The raw text to split

        Returns:
            list of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        return splitter.split_text(text)

    def create_documents(
        self, text: str, metadata: dict[str, Any] = None
    ) -> list[Document]:
        """
        Create Document objects from chunked text.

        Args:
            text: The raw text to split
            metadata: Base metadata to apply to all documents

        Returns:
            list of Document objects
        """
        chunks = self.chunk_text(text)
        documents = []

        base_metadata = metadata or {}

        for i, chunk in enumerate(chunks):
            # Copy base metadata and add chunk-specific info
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = f"chunk_{i}"

            # Create document
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        return documents

    def chunk_document_by_section(
        self, text: str, section_detector: Callable[[str], list[dict[str, Any]]]
    ) -> list[Document]:
        """
        Split text by detected sections rather than fixed sizes.

        Args:
            text: The raw text to split
            section_detector: Function that identifies sections in the text

        Returns:
            list of Document objects
        """
        sections = section_detector(text)
        documents = []

        for i, section in enumerate(sections):
            section_text = section.get("text", "")
            if len(section_text) > self.chunk_size:
                # Further chunk large sections
                section_chunks = self.chunk_text(section_text)
                for j, chunk in enumerate(section_chunks):
                    metadata = section.get("metadata", {}).copy()
                    metadata["section_id"] = f"section_{i}"
                    metadata["chunk_id"] = f"chunk_{j}"
                    documents.append(Document(page_content=chunk, metadata=metadata))
            else:
                # Use section as is
                metadata = section.get("metadata", {}).copy()
                metadata["section_id"] = f"section_{i}"
                documents.append(Document(page_content=section_text, metadata=metadata))

        return documents

    @staticmethod
    def create_section_detector(
        section_headers: list[str],
    ) -> Callable[[str], list[dict[str, Any]]]:
        """
        Create a simple section detector based on headers.

        Args:
            section_headers: list of headers that indicate new sections

        Returns:
            Function that detects sections in text
        """

        def detect_sections(text: str) -> list[dict[str, Any]]:
            # Create regex pattern for the headers
            pattern = "|".join(map(re.escape, section_headers))

            # Find all occurrences of section headers
            matches = list(re.finditer(f"({pattern})", text))

            sections = []
            for i, match in enumerate(matches):
                start = match.start()
                # Calculate end (either next section or end of text)
                end = matches[i + 1].start() if i < len(matches) - 1 else len(text)

                header = match.group(0)
                section_text = text[start:end]

                sections.append(
                    {
                        "text": section_text,
                        "metadata": {
                            "header": header,
                            "start_pos": start,
                            "end_pos": end,
                        },
                    }
                )

            # Handle case where no sections are found
            if not sections and text:
                sections.append(
                    {
                        "text": text,
                        "metadata": {
                            "header": "Main",
                            "start_pos": 0,
                            "end_pos": len(text),
                        },
                    }
                )

            return sections

        return detect_sections
