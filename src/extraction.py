"""
src/extraction.py
------------------
Extracts text from a PDF.

For each page, it first tries native text extraction (fast, accurate, works
for normal digital PDFs). If a page has little or no extractable text - a
strong sign it's a scanned image rather than real text - it falls back to
OCR using Tesseract (free, open-source, runs entirely on your machine).

No cloud OCR service, no API key, no cost.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# If a page's native text is shorter than this, we assume it's a scanned
# image and OCR it instead.
MIN_CHARS_BEFORE_OCR_FALLBACK = 20


@dataclass
class PageResult:
    page_number: int
    text: str
    method: str  # "native" or "ocr"


@dataclass
class ExtractionResult:
    filename: str
    pages: list[PageResult] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def used_ocr(self) -> bool:
        return any(p.method == "ocr" for p in self.pages)


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """Render a single PDF page to an image and run Tesseract OCR on it."""
    images = convert_from_path(
        str(pdf_path), first_page=page_number, last_page=page_number, dpi=300
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def extract_text(pdf_path: str | Path) -> ExtractionResult:
    """
    Extract text from every page of a PDF, using OCR automatically for
    pages that don't contain real extractable text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No such file: {pdf_path}")

    result = ExtractionResult(filename=pdf_path.name)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            native_text = (page.extract_text() or "").strip()

            if len(native_text) >= MIN_CHARS_BEFORE_OCR_FALLBACK:
                result.pages.append(
                    PageResult(page_number=i, text=native_text, method="native")
                )
            else:
                ocr_text = _ocr_page(pdf_path, i).strip()
                result.pages.append(
                    PageResult(page_number=i, text=ocr_text, method="ocr")
                )

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/extraction.py <path_to_pdf>")
        sys.exit(1)

    res = extract_text(sys.argv[1])
    print(f"Extracted {len(res.pages)} page(s) from {res.filename}")
    print(f"OCR used: {res.used_ocr}")
    print("\n--- Text preview ---\n")
    print(res.full_text[:1000])
