"""
Phase I – Metadata attachment.

Enriches each Segment's SegmentMetadata with any information that
can be derived post-segmentation (e.g. absolute character offsets
within the source document, resolved parent headings).

The paper specifies metadata fields:
  filename, format, section title, page number, nesting depth, parent heading.
All of these are set during segmentation; this module handles the
offset computation which requires the original cleaned text.
"""

from __future__ import annotations

from typing import List

from model_bundle.schema import Segment


def attach_metadata(segments: List[Segment], cleaned_text: str) -> List[Segment]:
    """
    Resolve character offsets (char_start, char_end) for each segment
    within the cleaned source text.

    Offsets are needed by E-QuaRUM for fine-grained evidence alignment.
    We perform a sequential scan so overlapping chunks get consistent anchors.
    """
    search_start = 0
    for seg in segments:
        idx = cleaned_text.find(seg.text[:80], search_start)
        if idx == -1:
            # Fallback: search from document beginning
            idx = cleaned_text.find(seg.text[:80])
        if idx != -1:
            seg.metadata.char_start = idx
            seg.metadata.char_end = idx + len(seg.text)
            # Advance search cursor slightly before end to handle overlaps
            search_start = max(search_start, idx + len(seg.text) - 50)

    return segments
