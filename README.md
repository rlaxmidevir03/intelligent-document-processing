# Intelligent Document Processing 📄

This project takes a PDF and actually *understands* it — it pulls out the
text, figures out what kind of document it is, writes a short summary, and
lets you ask questions about it in plain English.

It's built to run entirely on your own machine. No cloud OCR service, no
paid AI API, no API keys anywhere in this repo. The only "cost" is that
the AI models get downloaded once from Hugging Face the first time you
run it (they're free, open-source, and get cached locally after that).

## What it does

1. **Extracts text from a PDF** — for normal PDFs it reads the text
   directly. For scanned documents (just an image, no real text), it
   automatically falls back to OCR using Tesseract.
2. **Classifies the document** — figures out if it's an invoice,
   contract, resume, receipt, etc., using a zero-shot model (meaning you
   can describe categories in plain English and it just works).
3. **Summarizes it** — generates a short summary of the content.
4. **Answers questions about it** — ask something like "what's the total
   amount due?" and it'll find the answer in the text.

## What's in this project

- **`src/extraction.py`** – pulls text out of a PDF, OCR fallback included
- **`src/classification.py`** – figures out the document type
- **`src/summarization.py`** – generates the summary
- **`src/qa.py`** – answers questions about the document
- **`src/pipeline.py`** – runs extraction → classification → summarization
  together
- **`api/main.py`** – a FastAPI app with `/process` and `/ask` endpoints
- **`demo_app.py`** – a Streamlit app where you can upload a PDF and try
  everything through a simple UI
- **`sample_docs/`** – a couple of sample PDFs to test with (one normal,
  one scanned-style) so you don't need your own file to try it out
- **`tests/`** – tests for the extraction, classification, summarization,
  QA, and API pieces

## How to run it

You'll need two system tools installed first (both free):

- **Tesseract** (OCR engine) — on Mac: `brew install tesseract`, on
  Ubuntu/Debian: `sudo apt install tesseract-ocr`, on Windows: [download
  here](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler** (renders PDF pages as images for OCR) — on Mac:
  `brew install poppler`, on Ubuntu/Debian: `sudo apt install
  poppler-utils`, on Windows: [download here](https://github.com/oschwartz10612/poppler-windows)

Then:

```bash
git clone https://github.com/<your-username>/intelligent-document-processing.git
cd intelligent-document-processing

pip install -r requirements.txt
```

Try it on a sample file straight from the command line:

```bash
python src/pipeline.py sample_docs/sample_text.pdf
```

Or run the API:

```bash
uvicorn api.main:app --reload
```
Then open `http://127.0.0.1:8000/docs` to upload a PDF and try it from
the browser.

Or run the interactive demo instead:

```bash
streamlit run demo_app.py
```

Heads up: the first time you run any of these, it'll download the AI
models (a few hundred MB total) — that part needs internet access, but
it's a one-time thing and it's completely free.

## Running the tests

```bash
pytest -v
```

The extraction tests always run. The classification/summarization/QA/API
tests are skipped automatically if you haven't installed the ML
dependencies yet, so the test suite still runs cleanly either way.

## Notes

- This is a portfolio/learning project. It's genuinely useful for basic
  document understanding, but it isn't meant to replace a production
  document-processing system for things like legal or financial
  compliance.
- Everything runs locally by default. If you wanted to deploy this
  somewhere (like AWS), the FastAPI service is already set up to run in
  Docker, so that part's a natural next step — just optional, and not
  required to use or demo the project.
