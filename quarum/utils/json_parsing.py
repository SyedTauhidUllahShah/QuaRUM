"""
JSON extraction and parsing utilities.

This module provides utilities for extracting and parsing JSON from
LLM responses, handling common issues and edge cases.
"""

import re
import json
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class JSONExtractor:
    """
    Extracts and parses JSON from LLM responses.

    This class handles various edge cases and formats that may occur
    in JSON returned by language models, including markdown formatting,
    incomplete JSON, and syntax errors.
    """

    def __init__(self):
        """Initialize the JSON extractor."""
        self.last_error = None

    def extract(self, text: str) -> Optional[dict[str, Any]]:
        """
        Extract a JSON object from text.

        Args:
            text: Text containing JSON

        Returns:
            Parsed JSON object or None if extraction failed
        """
        self.last_error = None

        try:
            # Clean the text to find JSON
            cleaned_text = self._clean_text(text)

            # Find JSON blocks
            json_blocks = self._find_json_blocks(cleaned_text)
            if not json_blocks:
                self.last_error = "No JSON object found in response"
                logger.warning(self.last_error)
                return None

            # Get the largest JSON block
            json_str = max(json_blocks, key=len)

            # Fix common JSON syntax issues
            json_str = self._fix_json_syntax(json_str)

            # Parse JSON
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            self.last_error = f"Failed to parse JSON: {str(e)}"
            logger.warning(self.last_error)
            return None
        except Exception as e:
            self.last_error = f"Error extracting JSON: {str(e)}"
            logger.warning(self.last_error)
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing markdown formatting.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove markdown code block markers
        cleaned = re.sub(r"```(?:json)?\n|```", "", text)

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def _find_json_blocks(self, text: str) -> list[str]:
        """
        Find JSON blocks in text.

        Args:
            text: Cleaned text

        Returns:
            list of potential JSON blocks
        """
        # Look for objects with curly braces
        json_blocks = re.findall(r"\{[\s\S]*?\}", text)

        # Also look for arrays with square brackets
        json_arrays = re.findall(r"\[[\s\S]*?\]", text)

        # Combine and return all potential JSON blocks
        return json_blocks + json_arrays

    def _fix_json_syntax(self, json_str: str) -> str:
        """
        Fix common JSON syntax issues.

        Args:
            json_str: JSON string with potential issues

        Returns:
            Fixed JSON string
        """
        # Fix trailing commas before closing brackets
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        # Fix missing quotes around property names
        json_str = re.sub(r"(\{|\,)\s*([a-zA-Z0-9_]+)\s*:", r'\1"\2":', json_str)

        # Fix single quotes used instead of double quotes
        # This is more complex and can lead to issues, so we'll only do basic fixes
        if '"' not in json_str and "'" in json_str:
            json_str = json_str.replace("'", '"')

        return json_str

    def extract_multiple(self, text: str) -> list[dict[str, Any]]:
        """
        Extract multiple JSON objects from text.

        Args:
            text: Text containing multiple JSON objects

        Returns:
            list of parsed JSON objects
        """
        self.last_error = None
        results = []

        try:
            # Clean the text
            cleaned_text = self._clean_text(text)

            # Find all JSON blocks
            json_blocks = self._find_json_blocks(cleaned_text)

            # Process each block
            for json_str in json_blocks:
                try:
                    # Fix syntax issues
                    fixed_json = self._fix_json_syntax(json_str)

                    # Parse JSON
                    json_obj = json.loads(fixed_json)

                    # Add to results
                    results.append(json_obj)
                except json.JSONDecodeError:
                    # Skip invalid blocks
                    continue

            return results

        except Exception as e:
            self.last_error = f"Error extracting multiple JSON objects: {str(e)}"
            logger.warning(self.last_error)
            return results

    def extract_type(
        self, text: str, expected_type: type
    ) -> Optional[Union[dict[str, Any], list[Any]]]:
        """
        Extract JSON and verify it matches expected type.

        Args:
            text: Text containing JSON
            expected_type: Expected type (dict or list)

        Returns:
            Parsed JSON if it matches expected type, None otherwise
        """
        result = self.extract(text)

        if result is None:
            return None

        if not isinstance(result, expected_type):
            self.last_error = (
                f"Extracted JSON is not of expected type {expected_type.__name__}"
            )
            logger.warning(self.last_error)
            return None

        return result

    def extract_with_schema(
        self, text: str, required_keys: list[str]
    ) -> Optional[dict[str, Any]]:
        """
        Extract JSON and verify it contains required keys.

        Args:
            text: Text containing JSON
            required_keys: list of required keys

        Returns:
            Parsed JSON if it contains all required keys, None otherwise
        """
        result = self.extract(text)

        if result is None or not isinstance(result, dict):
            return None

        # Check for required keys
        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            self.last_error = (
                f"Extracted JSON is missing required keys: {', '.join(missing_keys)}"
            )
            logger.warning(self.last_error)
            return None

        return result
