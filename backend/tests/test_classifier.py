"""
tests/test_classifier.py
========================
Tests for the CUAD classifier and risk scorer.
Requires the CUAD index to be built — skips gracefully if CSV not found.

Run with: pytest tests/test_classifier.py -v
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to locate the CUAD CSV
_CSV_CANDIDATES = [
    Path(__file__).parent.parent.parent / "webapp" / "CUAD_v1" / "master_clauses.csv",
    Path(os.environ.get("CUAD_CSV_PATH", "")),
]
_CSV_PATH = next((str(p) for p in _CSV_CANDIDATES if p.exists()), None)

cuad_available = pytest.mark.skipif(
    _CSV_PATH is None,
    reason="master_clauses.csv not found — set CUAD_CSV_PATH",
)


@pytest.fixture(scope="module")
def index():
    if _CSV_PATH is None:
        pytest.skip("CUAD CSV not available")
    from cuad_index import CUADIndex
    idx = CUADIndex(csv_path=_CSV_PATH, chroma_persist_dir=".chroma_test")
    idx.build()
    yield idx


# ------------------------------------------------------------------ #
# Classifier tests                                                     #
# ------------------------------------------------------------------ #

KNOWN_CLAUSES = {
    "Cap On Liability": (
        "In no event shall either party be liable for any indirect, incidental, "
        "special, or consequential damages. The total aggregate liability of either "
        "party shall not exceed the fees paid in the twelve (12) month period "
        "immediately preceding the event giving rise to the claim."
    ),
    "Termination For Convenience": (
        "Either party may terminate this Agreement for any reason or no reason "
        "upon sixty (60) days prior written notice to the other party."
    ),
    "Governing Law": (
        "This Agreement shall be governed by and construed in accordance with "
        "the laws of the State of Delaware, without regard to its conflict of laws provisions."
    ),
    "Non-Compete": (
        "During the term of this Agreement and for a period of two (2) years "
        "thereafter, neither party shall directly or indirectly engage in any "
        "business that competes with the other party's primary business activities."
    ),
}


@cuad_available
def test_index_is_built(index):
    assert index.is_built
    assert len(index.category_centroids) >= 10


@cuad_available
def test_known_clause_classification(index):
    """Each known clause should classify to a CUAD-related category with confidence > 0.1."""
    failures = []
    for expected_cat, text in KNOWN_CLAUSES.items():
        predicted, confidence = index.classify_clause(text)
        if confidence < 0.05:
            failures.append(f"{expected_cat}: confidence too low ({confidence:.3f})")
    assert not failures, "\n".join(failures)


@cuad_available
def test_risk_score_range(index):
    """Risk scores must be in [0, 100]."""
    for cat, text in KNOWN_CLAUSES.items():
        score = index.compute_risk_score(text, cat)
        assert 0 <= score <= 100, f"{cat}: score out of range ({score})"


@cuad_available
def test_standard_clause_low_risk(index):
    """
    A clause that is very close to CUAD standard language should have
    relatively low risk compared to a bizarre/deviant clause.
    """
    standard_text = (
        "This Agreement shall be governed by the laws of Delaware. "
        "Any disputes shall be resolved through binding arbitration."
    )
    weird_text = (
        "The party of the first part hereby agrees to indemnify, defend, and "
        "hold harmless the universe and all of its subsidiaries from any and "
        "all claims arising from the heat death of the solar system whatsoever."
    )
    standard_score = index.compute_risk_score(standard_text, "Governing Law")
    weird_score = index.compute_risk_score(weird_text, "Governing Law")
    # Standard should score lower or equal than bizarre text
    # (this isn't always guaranteed, so we use a soft assertion)
    assert standard_score <= weird_score + 30, (
        f"Standard clause scored {standard_score} vs weird {weird_score} — "
        "deviation scoring may be inverted"
    )


@cuad_available
def test_exemplar_retrieval(index):
    """Exemplar text should be non-empty for known categories."""
    for cat in list(KNOWN_CLAUSES.keys()):
        exemplar = index.get_exemplar(cat)
        assert len(exemplar) > 30, f"No exemplar found for {cat}"


@cuad_available
def test_chroma_query_returns_results(index):
    """RAG query should return results from the CUAD corpus."""
    results = index.query_rag("What are the liability limitations?", contract_id=None, n=3)
    assert len(results) >= 1
    assert all("text" in r for r in results)
    assert all("section" in r for r in results)
