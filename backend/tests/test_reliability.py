"""
tests/test_reliability.py
==========================
Reliability and graceful failure handling tests.
"""
import os
import sys
import pytest
import httpx
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.parser import extract_text, TextExtractionError, OCRError
from services.rag import answer_question
from cuad_index import CUADIndex


# ------------------------------------------------------------------ #
# Extraction / OCR failure tests                                       #
# ------------------------------------------------------------------ #

def test_scanned_pdf_without_ocr_libraries():
    """If pdfminer returns short text and OCR dependencies are missing, raise TextExtractionError."""
    mock_high_level = MagicMock()
    mock_high_level.extract_text.return_value = "   "
    
    # Ensure they raise ImportError on import by mapping to None in sys.modules
    with patch.dict("sys.modules", {
        "pdfminer.high_level": mock_high_level,
        "pdf2image": None,
        "pytesseract": None
    }):
        with pytest.raises(TextExtractionError) as exc_info:
            extract_text(b"some bytes", "scanned_document.pdf")
        
        assert "OCR dependencies" in str(exc_info.value)
        assert "not installed" in str(exc_info.value)


def test_ocr_conversion_to_images_fails():
    """If converting PDF to images raises an exception, raise OCRError."""
    mock_high_level = MagicMock()
    mock_high_level.extract_text.return_value = ""
    
    # Mock pdf2image to raise a RuntimeError
    mock_pdf2image = MagicMock()
    mock_pdf2image.convert_from_bytes.side_effect = RuntimeError("Poppler not found")
    mock_pytesseract = MagicMock()
    
    # Inject mocked modules into sys.modules
    with patch.dict("sys.modules", {
        "pdfminer.high_level": mock_high_level,
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract
    }):
        with pytest.raises(OCRError) as exc_info:
            extract_text(b"%PDF-1.4 ...", "scanned_document.pdf")
        
        assert "Poppler" in str(exc_info.value)
        assert "OCR failed" in str(exc_info.value)


def test_ocr_tesseract_fails():
    """If tesseract throws an execution error, raise OCRError."""
    mock_high_level = MagicMock()
    mock_high_level.extract_text.return_value = ""
    
    # Mock pdf2image to return a list of dummy images
    mock_pdf2image = MagicMock()
    dummy_image = MagicMock()
    mock_pdf2image.convert_from_bytes.return_value = [dummy_image]
    
    # Mock pytesseract to raise an exception
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.side_effect = RuntimeError("Tesseract not found in PATH")
    
    # Inject mocked modules into sys.modules
    with patch.dict("sys.modules", {
        "pdfminer.high_level": mock_high_level,
        "pdf2image": mock_pdf2image,
        "pytesseract": mock_pytesseract
    }):
        with pytest.raises(OCRError) as exc_info:
            extract_text(b"%PDF-1.4 ...", "scanned.pdf")
        
        assert "Tesseract" in str(exc_info.value)
        assert "OCR failed" in str(exc_info.value)


# ------------------------------------------------------------------ #
# LLM / Assistant RAG failure tests                                   #
# ------------------------------------------------------------------ #

def test_llm_timeout_graceful_degradation():
    """If the LLM API call times out, the assistant falls back to template answer with degraded banner."""
    # Setup mock index
    mock_index = MagicMock()
    mock_index._collection = MagicMock()
    mock_index._collection.count.return_value = 10
    
    # Mock retrieved clauses from vector search
    mock_clauses = [
        {"section": "Section 1", "text": "This is liability clause.", "category": "Cap On Liability", "score": 0.9}
    ]
    mock_index.query_rag.return_value = mock_clauses
    
    # Mock environment variable to simulate GEMINI_API_KEY being present
    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeKey"}):
        # Mock httpx.Client to raise a TimeoutException
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
            mock_client_cls.return_value.__enter__.return_value = mock_client
            
            result = answer_question(
                question="What is the liability cap?",
                contract_id="test_contract",
                index=mock_index,
                contract_name="Test Agreement"
            )
            
            assert "degraded mode" in result["content"]
            assert "timeout/connection failure" in result["content"]
            # Verify the direct contract clause extraction fallback is present
            assert "Section 1" in result["content"]
            assert "liability clause" in result["content"]
            assert len(result["citations"]) == 1


def test_vector_search_failure_graceful_handling():
    """If ChromaDB query throws an exception, answer_question returns a graceful error message instead of crashing."""
    mock_index = MagicMock()
    mock_index._collection = MagicMock()
    mock_index._collection.count.return_value = 10
    # Simulate database crash/failure on query
    mock_index.query_rag.side_effect = Exception("ChromaDB connection broken")
    
    result = answer_question(
        question="What is the liability cap?",
        contract_id="test_contract",
        index=mock_index,
        contract_name="Test Agreement"
    )
    
    assert "search engine is temporarily unavailable" in result["content"]
    assert len(result["citations"]) == 0
