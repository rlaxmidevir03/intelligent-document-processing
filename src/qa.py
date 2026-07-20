"""
src/qa.py
---------
Answers questions about a document using a local extractive
question-answering model (DistilBERT fine-tuned on SQuAD). Given the
document's text and a question, it finds and returns the most relevant
span of text as the answer - entirely offline, no API key needed.
"""

from functools import lru_cache

MODEL_NAME = "distilbert-base-cased-distilled-squad"

# Extractive QA models have a limited context window. If the document is
# longer than this, we search it in overlapping windows and keep the
# highest-confidence answer.
MAX_CONTEXT_CHARS = 3000
CONTEXT_OVERLAP_CHARS = 200


@lru_cache(maxsize=1)
def _get_qa_pipeline():
    from transformers import pipeline

    return pipeline("question-answering", model=MODEL_NAME)


def _chunk_context(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def answer_question(text: str, question: str) -> dict:
    """
    Find the answer to `question` within `text`.
    Returns the best-scoring answer across all context windows.
    """
    if not text.strip():
        raise ValueError("Cannot search an empty document.")
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    qa = _get_qa_pipeline()
    chunks = _chunk_context(text, MAX_CONTEXT_CHARS, CONTEXT_OVERLAP_CHARS)

    best_result = None
    for chunk in chunks:
        result = qa(question=question, context=chunk)
        if best_result is None or result["score"] > best_result["score"]:
            best_result = result

    return {
        "answer": best_result["answer"],
        "confidence": round(best_result["score"], 4),
    }


if __name__ == "__main__":
    sample = (
        "Invoice Number: INV-2026-0042. Amount Due: $4,250.00. "
        "Due Date: August 1, 2026. Bill To: Acme Corporation."
    )
    print(answer_question(sample, "How much is due?"))
    print(answer_question(sample, "Who is this invoice billed to?"))
