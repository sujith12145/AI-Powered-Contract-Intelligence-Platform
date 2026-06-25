"""
services/parser.py
==================
Parses uploaded contract files (PDF, DOCX, TXT) into:
  - raw text
  - list of clause segments: [{section, text}]

Clause segmentation uses a two-pass strategy:
  1. Look for numbered section headers (e.g. "1.", "Article 1", "SECTION 2.3")
  2. Fall back to paragraph chunking with a minimum length threshold
"""
from __future__ import annotations

import io
import re
from typing import List, Dict, Optional

# ------------------------------------------------------------------ #
# PDF parsing                                                          #
# ------------------------------------------------------------------ #

def parse_pdf(data: bytes) -> str:
    """Extract plain text from a PDF file's bytes."""
    try:
        from pdfminer.high_level import extract_text as _extract
        text = _extract(io.BytesIO(data))
        return text or ""
    except Exception as exc:
        raise ValueError(f"PDF parsing failed: {exc}") from exc


# ------------------------------------------------------------------ #
# DOCX parsing                                                         #
# ------------------------------------------------------------------ #

def parse_docx(data: bytes) -> str:
    """Extract plain text from a DOCX file's bytes."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise ValueError(f"DOCX parsing failed: {exc}") from exc


# ------------------------------------------------------------------ #
# Text normalisation                                                    #
# ------------------------------------------------------------------ #

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ------------------------------------------------------------------ #
# Section header detection                                             #
# ------------------------------------------------------------------ #

# Patterns for common legal section headers:
#   "1."  "1.1"  "Article 1"  "SECTION 2.3"  "Section 4 –"
_SECTION_RE = re.compile(
    r"(?m)^("
    r"(?:ARTICLE|SECTION|CLAUSE|PART|EXHIBIT|SCHEDULE|APPENDIX)\s+[\dA-Z]+[.\s]"
    r"|"
    r"\d+(?:\.\d+)*[.\)]\s"
    r")",
    re.IGNORECASE,
)


def _split_by_headers(text: str) -> List[Dict[str, str]]:
    """Split text at detected section headers. Returns [{section, text}]."""
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 2:
        return []

    segments: List[Dict[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_label = match.group(0).strip().rstrip(".)")
        body = text[start:end].strip()
        if len(body) > 80:
            segments.append({"section": section_label, "text": body})
    return segments


def _split_by_paragraphs(text: str, min_chars: int = 150) -> List[Dict[str, str]]:
    """
    Fall-back: split on blank lines, label each paragraph sequentially.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= min_chars]
    segments: List[Dict[str, str]] = []
    for i, para in enumerate(paras):
        segments.append({"section": f"¶{i+1}", "text": para})
    return segments


def _merge_short_segments(
    segments: List[Dict[str, str]], min_chars: int = 100
) -> List[Dict[str, str]]:
    """Merge segments shorter than min_chars into the previous one."""
    merged: List[Dict[str, str]] = []
    for seg in segments:
        if merged and len(seg["text"]) < min_chars:
            merged[-1]["text"] += "\n" + seg["text"]
        else:
            merged.append(dict(seg))
    return merged


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def extract_text(data: bytes, filename: str) -> str:
    """
    Dispatch to the appropriate parser based on file extension.
    Returns raw text string.
    """
    name_lower = filename.lower()
    if name_lower.endswith(".pdf"):
        return parse_pdf(data)
    elif name_lower.endswith((".docx", ".doc")):
        return parse_docx(data)
    else:
        # Assume plain text
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""


def segment_clauses(text: str, max_clauses: int = 30) -> List[Dict[str, str]]:
    """
    Segment contract text into clause-level chunks.

    Returns a list of dicts:
        {"section": "Section 2.1", "text": "...full clause text..."}

    At most `max_clauses` segments are returned (longer contracts are
    truncated at the paragraph-split stage to keep analysis fast).
    """
    text = _normalise(text)

    # Try header-based splitting first
    segments = _split_by_headers(text)
    if len(segments) < 3:
        # Fall back to paragraph splitting
        segments = _split_by_paragraphs(text)

    # Merge tiny fragments
    segments = _merge_short_segments(segments)

    # Cap
    return segments[:max_clauses]


def infer_contract_metadata(text: str, filename: str) -> Dict[str, str]:
    """
    Best-effort extraction of basic contract metadata from raw text.
    Returns a dict with keys: counterparty, effectiveDate, expiryDate, type, value.
    """
    meta: Dict[str, str] = {
        "counterparty": "Unknown Party",
        "effectiveDate": "",
        "expiryDate": "",
        "value": "N/A",
        "type": _guess_contract_type(filename, text),
    }

    # Effective date
    date_match = re.search(
        r"(?:effective|entered into|dated?)[^\n]*?(\b(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        text[:10000],
        re.IGNORECASE,
    )
    if date_match:
        meta["effectiveDate"] = date_match.group(1)

    # Counterparty — look for company names with suffixes in the first 10000 chars
    # Uses backward matching from suffix anchors, filtering out state-of-incorporation boilerplate
    clean_text = re.sub(r'\s+', ' ', text[:10000]).strip()
    suffixes = [
        r'\bInc\b\.?', r'\bLLC\b', r'\bCorp\b\.?', r'\bLtd\b\.?', r'\bLP\b', r'\bLLP\b', r'\bCo\b\.?', 
        r'\bCorporation\b', r'\bLimited\b', r'\bSolutions\b', r'\bLogistics\b', r'\bSystems\b', 
        r'\bTechnologies\b', r'\bPartners\b', r'\bHoldings\b', r'\bGroup\b', r'\bBank\b'
    ]
    
    companies = []
    suffix_pat = '|'.join(suffixes)
    
    for match in re.finditer(suffix_pat, clean_text, re.IGNORECASE):
        start_pos = match.start()
        preceding = clean_text[max(0, start_pos-50):start_pos]
        words_match = re.search(r'\b([A-Z][a-zA-Z0-9&\'-]*(?:\s+[A-Z][a-zA-Z0-9&\'-]*){0,4})\b(?:\s*,\s*)?\s*$', preceding)
        if words_match:
            full_name = words_match.group(1) + " " + match.group(0)
            full_name = full_name.strip().rstrip(',. ')
            full_name = re.sub(r'^(?:this|agreement|party|parties|contract|schedule|exhibit|section|article|by|and|between)\s+', '', full_name, flags=re.I)
            
            # Filter out common state-of-incorporation boilerplate descriptions
            lower_name = full_name.lower()
            if "delaware" in lower_name and any(x in lower_name for x in ["corporation", "limited", "llc", "company"]):
                continue
            if "california" in lower_name and any(x in lower_name for x in ["corporation", "limited", "llc", "company"]):
                continue
            if any(state in lower_name for state in ["new york", "texas", "florida", "state of", "commonwealth of"]):
                if any(x in lower_name for x in ["corporation", "limited", "llc", "company"]):
                    continue
                    
            if full_name and full_name not in companies:
                companies.append(full_name)
                
    # Sort companies by length descending to make substring filtering easy
    companies.sort(key=len, reverse=True)
    
    # Filter out substrings (e.g. "Brightline Logistics" is sub of "Brightline Logistics Inc")
    unique_companies = []
    for c in companies:
        is_sub = False
        for other in unique_companies:
            if c in other:
                is_sub = True
                break
        if not is_sub:
            unique_companies.append(c)
            
    # Sort back by order of appearance in text
    unique_companies.sort(key=lambda x: clean_text.find(x))
    
    if len(unique_companies) >= 2:
        meta["counterparty"] = unique_companies[1]
    elif len(unique_companies) == 1:
        meta["counterparty"] = unique_companies[0]
    else:
        # Fallback to original regex
        between_match = re.search(
            r'between\s+([A-Z][A-Za-z\s,\.]+(?:Inc\.|LLC|Corp\.|Ltd\.|LP|LLP|Co\.|Corporation|Limited)?)'
            r'[\s,]*(\([^)]*\))?[\s,]*'
            r'and\s+([A-Z][A-Za-z\s,\.]+(?:Inc\.|LLC|Corp\.|Ltd\.|LP|LLP|Co\.|Corporation|Limited)?)',
            clean_text,
            re.IGNORECASE,
        )
        if between_match:
            counterparty = between_match.group(3).strip().rstrip(',. "')
            counterparty = re.sub(r'\s*\(.*$', '', counterparty).strip()
            if counterparty:
                meta["counterparty"] = counterparty
 
     # Dollar value
    value_match = re.search(
        r"\$\s?([\d,]+(?:\.\d{2})?)\s*(?:million|thousand|USD|dollars)?",
        text[:10000],
        re.IGNORECASE,
    )
    if value_match:
        meta["value"] = f"${value_match.group(1)}"
 
    return meta


_CONTRACT_TYPE_KEYWORDS = {
    "NDA": ["non-disclosure", "nda", "confidential"],
    "SaaS License": ["saas", "software as a service", "cloud service"],
    "License Agreement": ["license agreement", "licence agreement"],
    "Vendor Agreement": ["vendor", "supplier", "services agreement"],
    "Employment Contract": ["employment", "employee", "employer"],
    "Distribution Agreement": ["distribution", "reseller", "channel partner"],
    "Procurement Contract": ["procurement", "purchase order", "supply"],
    "Service Agreement": ["service agreement", "professional services"],
    "Consulting Agreement": ["consulting", "consultant"],
    "Joint Venture": ["joint venture", "collaboration agreement"],
}


def _guess_contract_type(filename: str, text: str) -> str:
    combined = (filename + " " + text[:500]).lower()
    for ctype, keywords in _CONTRACT_TYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return ctype
    return "Commercial Agreement"
