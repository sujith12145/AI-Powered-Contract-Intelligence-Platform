"""
tests/test_parser.py
====================
Tests for the contract text parser / segmenter.
Run with: pytest tests/test_parser.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.parser import segment_clauses, infer_contract_metadata, _guess_contract_type

SAMPLE_NDA = """
MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of January 15, 2024,
between Acme Corporation ("Disclosing Party") and Beta Inc. ("Receiving Party").

1. Confidentiality Obligations

Each party agrees to hold the other party's Confidential Information in strict confidence
and shall not disclose such information to any third party for a period of two (2) years
from the date of disclosure, without the prior written consent of the disclosing party.
This obligation shall survive termination of this Agreement.

2. Intellectual Property

All inventions and works created jointly by the parties shall be jointly owned.
Each party shall have the right to exploit such IP for commercial purposes without
accounting to the other party.

3. Term and Termination

This Agreement shall remain in effect for a period of two (2) years from the Effective Date.
Either party may terminate upon thirty (30) days written notice. Confidentiality obligations
shall survive termination for a period of three (3) years.

4. Governing Law

This Agreement shall be governed by the laws of the State of Delaware.
"""


def test_segment_clauses_finds_sections():
    segments = segment_clauses(SAMPLE_NDA)
    assert len(segments) >= 3, f"Expected ≥3 segments, got {len(segments)}"


def test_segment_clauses_section_labels():
    segments = segment_clauses(SAMPLE_NDA)
    labels = [s["section"] for s in segments]
    # Should find numbered sections
    assert any("1" in lbl or "2" in lbl for lbl in labels), f"No numbered sections: {labels}"


def test_segment_clauses_text_not_empty():
    segments = segment_clauses(SAMPLE_NDA)
    for seg in segments:
        assert len(seg["text"]) > 50, f"Segment too short: {seg}"


def test_segment_clauses_max_limit():
    long_text = "\n\n".join([f"{i}. Clause text here that is long enough " * 5 for i in range(50)])
    segments = segment_clauses(long_text, max_clauses=15)
    assert len(segments) <= 15


def test_infer_metadata_type():
    meta = infer_contract_metadata(SAMPLE_NDA, "mutual_nda.pdf")
    assert meta["type"] == "NDA"


def test_infer_metadata_counterparty():
    meta = infer_contract_metadata(SAMPLE_NDA, "contract.pdf")
    assert "Beta" in meta["counterparty"] or meta["counterparty"] != "Unknown Party"


def test_guess_contract_type_saas():
    assert _guess_contract_type("saas_agreement.pdf", "software as a service") == "SaaS License"


def test_guess_contract_type_employment():
    assert _guess_contract_type("employment.pdf", "employment agreement between employer") == "Employment Contract"


def test_paragraph_fallback():
    """Documents without section numbers should still be segmented."""
    no_headers = "\n\n".join([
        "The parties agree that all information shared under this agreement "
        "shall be kept confidential for a period of two years and shall not "
        "be disclosed to any third party without prior written consent.",
        "Either party may terminate this agreement upon sixty days written notice "
        "to the other party. Upon termination, all obligations of confidentiality "
        "shall survive for a period of one year following termination.",
    ])
    segments = segment_clauses(no_headers)
    assert len(segments) >= 1
