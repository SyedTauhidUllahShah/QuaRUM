"""
Phase I – Segmentation.

Produces sentence-safe units that align with headings, section transitions,
paragraphs, lists, and tables.

Rules (from paper):
  - Target 350–600 tokens per segment; cap near 800
  - Keep bulleted and numbered lists intact
  - Do not split sentences
  - Do not cross heading boundaries
  - Add small overlaps so cues flow across boundaries (overlap=50 chars)
  - Avoid very small segments (< 50 tokens)

Implementation uses LangChain RecursiveCharacterTextSplitter (chunk_size=512,
overlap=50) as the paper's implementation table specifies, post-processed
to enforce heading boundaries and merge under-sized chunks.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from model_bundle.schema import Segment, SegmentMetadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_text(
    cleaned_text: str,
    doc_metadata: Dict,
) -> List[Segment]:
    """
    Split cleaned_text into Segment objects with attached metadata.

    Returns a list of Segment objects ordered by document position.
    """
    # 1. Extract section structure from the text
    sections = _split_into_sections(cleaned_text)

    # 2. For each section, further split if too long, preserving sentences
    segments: List[Segment] = []
    for section in sections:
        chunks = _split_section(section["text"])
        for chunk_text in chunks:
            if _token_count(chunk_text) < config.SEGMENT_MIN_TOKENS:
                continue  # discard very small segments
            seg = _make_segment(chunk_text, section, doc_metadata)
            segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

# Matches numbered headings: "2.", "2.1", "12.3", etc.
_HEADING_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})?\.?[ \t]+[A-Z].{2,80})$",   # [ \t]+ never crosses line boundaries
    re.MULTILINE,
)

# Matches subsection headings of format "N.N title" for nesting depth calculation
_SUBSECTION_RE = re.compile(r"^(\d+)\.(\d+)[ \t]")


def _split_into_sections(text: str) -> List[Dict]:
    """
    Split text at heading boundaries.
    Returns list of dicts: {text, section_title, nesting_depth, parent_heading}
    """
    # Find all heading positions
    heading_matches = list(_HEADING_RE.finditer(text))

    if not heading_matches:
        # No headings found – treat entire text as one section
        return [{"text": text, "section_title": None, "nesting_depth": 0, "parent_heading": None}]

    sections = []
    current_section_title = None
    current_parent = None
    current_depth = 0

    for i, match in enumerate(heading_matches):
        heading_text = match.group(1).strip()
        depth, parent = _compute_depth(heading_text)

        start = match.start()
        end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
        body = text[start:end].strip()

        if not body:
            continue

        sections.append({
            "text": body,
            "section_title": heading_text,
            "nesting_depth": depth,
            "parent_heading": parent,
        })

    # If there is text before the first heading, prepend it as a section
    if heading_matches and heading_matches[0].start() > 0:
        preamble = text[: heading_matches[0].start()].strip()
        if preamble and _token_count(preamble) >= config.SEGMENT_MIN_TOKENS:
            sections.insert(0, {
                "text": preamble,
                "section_title": None,
                "nesting_depth": 0,
                "parent_heading": None,
            })

    return sections


def _compute_depth(heading: str) -> Tuple[int, Optional[str]]:
    """Return (nesting_depth, parent_heading_number_prefix)."""
    m = _SUBSECTION_RE.match(heading)
    if m:
        # e.g. "2.1 Device Identity" -> depth 2, parent "2."
        return 2, f"{m.group(1)}."
    # Top-level: "2. Device Management" -> depth 1
    return 1, None


# ---------------------------------------------------------------------------
# Intra-section splitting
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.SEGMENT_CHUNK_SIZE,
    chunk_overlap=config.SEGMENT_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],   # prefer paragraph then sentence breaks
    keep_separator=True,
)


def _split_section(section_text: str) -> List[str]:
    """
    Split a single section's text into chunks.
    Merges under-sized chunks with the next chunk to avoid tiny segments.
    """
    raw_chunks = _splitter.split_text(section_text)
    if not raw_chunks:
        return []

    # Merge under-sized consecutive chunks
    merged: List[str] = []
    buffer = ""
    for chunk in raw_chunks:
        if buffer:
            candidate = buffer + " " + chunk
            if _token_count(buffer) < config.SEGMENT_MIN_TOKENS:
                buffer = candidate
                continue
        if buffer:
            merged.append(buffer.strip())
        buffer = chunk
    if buffer.strip():
        merged.append(buffer.strip())

    return merged


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------

def _make_segment(text: str, section: Dict, doc_metadata: Dict) -> Segment:
    seg_id = f"seg_{uuid.uuid4().hex[:12]}"
    meta = SegmentMetadata(
        segment_id=seg_id,
        source_document=doc_metadata.get("filename", "unknown"),
        format=doc_metadata.get("format", "txt"),
        section_title=section.get("section_title"),
        page_number=doc_metadata.get("page_count"),   # page_count used as proxy; per-page tracking needs richer parser
        nesting_depth=section.get("nesting_depth", 0),
        parent_heading=section.get("parent_heading"),
        token_count=_token_count(text),
    )
    return Segment(segment_id=seg_id, text=text, metadata=meta)


# ---------------------------------------------------------------------------
# Token counting (approximate: 1 token ≈ 4 characters)
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    return max(1, len(text) // 4)
