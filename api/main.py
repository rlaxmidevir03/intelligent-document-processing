"""
api/main.py
-----------
FastAPI service for the document processing pipeline.

Endpoints:
  POST /process   - upload a PDF, get extracted text + classification + summary
  POST /ask        - ask a question about a previously processed document
  GET  /health

Run locally (no keys, no paid services):
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.pipeline import process_document  # noqa: E402
from src.qa import answer_question  # noqa: E402

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Intelligent Document Processing API",
    description=(
        "Extracts, classifies, and summarizes PDF documents, and answers "
        "questions about them. Runs entirely on local, open-source models "
        "- no API key or paid service required."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store mapping a document_id to its extracted text, so /ask
# doesn't need to re-extract the PDF on every question. For a real
# production system this would be a database instead of a plain dict.
_document_store: dict[str, str] = {}


class ProcessResponse(BaseModel):
    document_id: str
    filename: str
    used_ocr: bool
    n_pages: int
    classification: dict | None
    summary: str | None
    text_preview: str
    warning: str | None = None


class AskRequest(BaseModel):
    document_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    confidence: float


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Intelligent Document Processing API is running.",
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse, tags=["Document"])
async def process(file: UploadFile = File(...)):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    document_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{document_id}.pdf"

    with saved_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        result = process_document(saved_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        saved_path.unlink(missing_ok=True)

    _document_store[document_id] = result.get("text", "")

    return ProcessResponse(
        document_id=document_id,
        filename=result["filename"],
        used_ocr=result["used_ocr"],
        n_pages=result["n_pages"],
        classification=result["classification"],
        summary=result["summary"],
        text_preview=result["text"][:500],
        warning=result.get("warning"),
    )


@app.post("/ask", response_model=AskResponse, tags=["Document"])
def ask(payload: AskRequest):
    text = _document_store.get(payload.document_id)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown document_id. Upload a document via /process first.",
        )
    try:
        result = answer_question(text, payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AskResponse(answer=result["answer"], confidence=result["confidence"])
