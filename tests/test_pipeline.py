"""
tests/test_pipeline.py
-----------------------
Tests for the document processing pipeline.

The extraction tests always run (they only need pdfplumber/pytesseract).
The classification/summarization/QA/API tests are skipped automatically
if torch/transformers aren't installed, so `pytest` still runs cleanly
in a minimal environment.

Run with:
    pytest -v
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.extraction import extract_text  # noqa: E402

SAMPLE_TEXT_PDF = ROOT_DIR / "sample_docs" / "sample_text.pdf"
SAMPLE_SCANNED_PDF = ROOT_DIR / "sample_docs" / "scanned_only.pdf"


def test_extract_native_text_pdf():
    if not SAMPLE_TEXT_PDF.exists():
        pytest.skip("Sample PDF not found.")
    result = extract_text(SAMPLE_TEXT_PDF)
    assert len(result.pages) >= 1
    assert not result.used_ocr
    assert "INVOICE" in result.full_text.upper()


def test_extract_scanned_pdf_uses_ocr():
    if not SAMPLE_SCANNED_PDF.exists():
        pytest.skip("Sample scanned PDF not found.")
    result = extract_text(SAMPLE_SCANNED_PDF)
    assert result.used_ocr
    assert len(result.full_text.strip()) > 0


def test_extract_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_text("does_not_exist.pdf")


def test_classification():
    pytest.importorskip("transformers")
    from src.classification import classify_document

    result = classify_document(
        "Invoice Number: 123. Amount Due: $500.", categories=["invoice", "resume"]
    )
    assert result["predicted_label"] in {"invoice", "resume"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_summarization():
    pytest.importorskip("transformers")
    from src.summarization import summarize_document

    text = (
        "The company reported strong quarterly earnings, with revenue up "
        "significantly year over year. Growth was driven by new product "
        "launches and expansion into international markets."
    )
    result = summarize_document(text, max_length=40, min_length=10)
    assert len(result["summary"]) > 0


def test_qa():
    pytest.importorskip("transformers")
    from src.qa import answer_question

    text = "The invoice amount due is $4,250.00, payable by August 1, 2026."
    result = answer_question(text, "How much is due?")
    assert "4,250" in result["answer"] or "4250" in result["answer"]


def test_api_process_and_ask():
    pytest.importorskip("transformers")
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    if not SAMPLE_TEXT_PDF.exists():
        pytest.skip("Sample PDF not found.")

    from api.main import app

    client = TestClient(app)

    with open(SAMPLE_TEXT_PDF, "rb") as f:
        resp = client.post(
            "/process", files={"file": ("sample_text.pdf", f, "application/pdf")}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] is not None

    ask_resp = client.post(
        "/ask", json={"document_id": body["document_id"], "question": "How much is due?"}
    )
    assert ask_resp.status_code == 200
