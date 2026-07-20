"""
src/classification.py
----------------------
Classifies a document's text into one of a set of categories, using a
zero-shot classification model from Hugging Face. "Zero-shot" means the
model wasn't specifically trained on your categories - you just describe
them in plain English (e.g. "invoice", "contract", "resume") and it works.

The model runs locally on your machine via PyTorch. The first time you run
it, it downloads the model weights from Hugging Face once (a few hundred
MB) and caches them - after that it works fully offline. No API key, no
per-request cost.
"""

from functools import lru_cache

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

DEFAULT_CATEGORIES = [
    "invoice",
    "contract",
    "resume",
    "receipt",
    "legal document",
    "email",
    "report",
    "letter",
]


@lru_cache(maxsize=1)
def _get_classifier():
    """Load the zero-shot classification pipeline once and reuse it."""
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=MODEL_NAME)


def classify_document(text: str, categories: list[str] | None = None) -> dict:
    """
    Classify `text` into one of `categories`.

    Returns a dict with the top predicted label and the full score
    breakdown across all candidate categories.
    """
    if not text.strip():
        raise ValueError("Cannot classify empty text.")

    categories = categories or DEFAULT_CATEGORIES
    classifier = _get_classifier()

    result = classifier(text, candidate_labels=categories, multi_label=False)

    return {
        "predicted_label": result["labels"][0],
        "confidence": round(result["scores"][0], 4),
        "all_scores": {
            label: round(score, 4)
            for label, score in zip(result["labels"], result["scores"])
        },
    }


if __name__ == "__main__":
    sample = (
        "Invoice Number: INV-2026-0042. Amount Due: $4,250.00. "
        "Due Date: August 1, 2026. Thank you for your business."
    )
    print(classify_document(sample))
