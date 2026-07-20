"""
src/pipeline.py
----------------
Runs the full document-understanding pipeline on a PDF:
extract text -> classify document type -> summarize.

QA is handled separately (see src/qa.py) since it depends on a specific
question, not something you'd run automatically on upload.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.classification import classify_document
from src.extraction import extract_text
from src.summarization import summarize_document


def process_document(pdf_path: str | Path, categories: list[str] | None = None) -> dict:
    """Run extraction, classification, and summarization on a single PDF."""
    extraction = extract_text(pdf_path)
    text = extraction.full_text

    if not text.strip():
        return {
            "filename": extraction.filename,
            "used_ocr": extraction.used_ocr,
            "n_pages": len(extraction.pages),
            "text": "",
            "classification": None,
            "summary": None,
            "warning": "No text could be extracted from this document.",
        }

    classification = classify_document(text, categories)
    summary = summarize_document(text)

    return {
        "filename": extraction.filename,
        "used_ocr": extraction.used_ocr,
        "n_pages": len(extraction.pages),
        "text": text,
        "classification": classification,
        "summary": summary["summary"],
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/pipeline.py <path_to_pdf>")
        sys.exit(1)

    result = process_document(sys.argv[1])
    # Trim full text in the printed preview so the terminal stays readable.
    preview = dict(result)
    preview["text"] = (preview["text"][:300] + "...") if preview["text"] else ""
    print(json.dumps(preview, indent=2))
