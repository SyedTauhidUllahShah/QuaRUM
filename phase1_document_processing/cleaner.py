"""
Phase I – Cleaning and normalisation.

Removes non-informative content using adaptive pattern detection:
only removes elements that recur across sections and are unlikely
to carry modeling value (headers, footers, page numbers, nav text).
Keeps useful content intact.

Handles the IoT dataset's specific artefacts:
  - Tab characters used as separators
  - Section headings concatenated directly into body text (PDF extraction)
  - Repeated document-level headers ("Smart Home IoT Control Hub | ...")
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(raw_text: str, fmt: str = "txt") -> str:
    """
    Clean and normalise raw extracted text.
    Returns cleaned text ready for segmentation.
    """
    text = raw_text

    # 1. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Replace tab characters with a single space (dataset-specific artefact)
    text = text.replace("\t", " ")

    # 3. Remove repeated document-level boilerplate lines
    text = _remove_recurring_lines(text)

    # 4. Insert line breaks before detected section headings
    #    (handles PDF extraction where headings run into body text)
    text = _insert_heading_breaks(text)

    # 5. Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Normalise whitespace within lines (multiple spaces -> single space)
    lines = [re.sub(r" {2,}", " ", line) for line in text.splitlines()]
    text = "\n".join(lines)

    # 7. Strip leading/trailing whitespace from the full document
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_recurring_lines(text: str) -> str:
    """
    Remove lines that appear verbatim more than once across the document
    AND are short enough to be boilerplate (< 80 chars).
    These are footers, page numbers, repeated headers, etc.
    """
    lines = text.splitlines()
    counts = Counter(line.strip() for line in lines if line.strip())

    cleaned = []
    seen_recurring = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if counts[stripped] > 1 and len(stripped) < 80:
            # Keep first occurrence, remove subsequent
            if stripped not in seen_recurring:
                seen_recurring.add(stripped)
                cleaned.append(line)
            # else: skip duplicate boilerplate
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


# Patterns for numbered section headings at various nesting levels:
#   "2. Device Management"
#   "2.1 Device Identity"
#   "12.3 Multi-User"
_HEADING_PATTERN = re.compile(
    r"(?<!\n)"                          # not already preceded by newline
    r"(?<![\d.])"                       # not preceded by digit OR period (avoids mid-number splits like "1.1")
    r"(\d{1,2}(?:\.\d{1,2})?[ \t]+[A-Z][A-Za-z])"  # heading number + space + Capital word
)


def _insert_heading_breaks(text: str) -> str:
    """
    Insert a newline before numbered section headings that were concatenated
    into body text during PDF/text extraction.
    e.g. "...last sentence.2. Device Management The device..."
         -> "...last sentence.\n2. Device Management The device..."
    """
    return _HEADING_PATTERN.sub(r"\n\1", text)
