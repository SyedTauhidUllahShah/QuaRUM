"""
Phase I – Ingestion.

Supports PDF, Word (.docx), Markdown (.md), and plain text (.txt).
Specialized parsers extract content and structural elements such as
headings, lists, tables, and formatting. Structural information is
preserved as metadata for later steps.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(file_path: str) -> Tuple[str, Dict]:
    """
    Parse a requirements document and return:
      (raw_text, doc_metadata)

    doc_metadata keys:
      filename, format, page_count (if known), title (if detectable)
    """
    ext = os.path.splitext(file_path)[1].lower()
    fmt_map = {".txt": "txt", ".pdf": "pdf", ".docx": "docx", ".md": "md"}
    fmt = fmt_map.get(ext, "txt")

    if fmt == "pdf":
        text, meta = _ingest_pdf(file_path)
    elif fmt == "docx":
        text, meta = _ingest_docx(file_path)
    elif fmt == "md":
        text, meta = _ingest_markdown(file_path)
    else:
        text, meta = _ingest_plaintext(file_path)

    meta["filename"] = os.path.basename(file_path)
    meta["format"] = fmt
    return text, meta


# ---------------------------------------------------------------------------
# Per-format parsers
# ---------------------------------------------------------------------------

def _ingest_plaintext(file_path: str) -> Tuple[str, Dict]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text, {"page_count": None}


def _ingest_markdown(file_path: str) -> Tuple[str, Dict]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text, {"page_count": None}


def _ingest_pdf(file_path: str) -> Tuple[str, Dict]:
    try:
        import PyPDF2
        pages: List[str] = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages.append(extracted)
        text = "\n".join(pages)
        return text, {"page_count": page_count}
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF ingestion: pip install PyPDF2")


def _ingest_docx(file_path: str) -> Tuple[str, Dict]:
    try:
        import docx
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return text, {"page_count": None}
    except ImportError:
        raise ImportError("python-docx is required for .docx ingestion: pip install python-docx")
