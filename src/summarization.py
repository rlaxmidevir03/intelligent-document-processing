"""
src/summarization.py
---------------------
Summarizes document text using a local transformer model (DistilBART,
distilled from Facebook's BART-CNN). Runs on your own machine - no API
key, no cost per request. The model is downloaded once from Hugging Face
and cached locally.
"""

from functools import lru_cache

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# The model has a limited input window, so very long documents are
# processed in chunks and the chunk summaries are combined.
MAX_CHUNK_CHARS = 3000


@lru_cache(maxsize=1)
def _get_summarizer():
    from transformers import pipeline

    return pipeline("summarization", model=MODEL_NAME)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks, current = [], []
    current_len = 0

    for word in words:
        current_len += len(word) + 1
        current.append(word)
        if current_len >= max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def summarize_document(
    text: str, max_length: int = 130, min_length: int = 30
) -> dict:
    """
    Summarize `text`. Long documents are split into chunks, summarized
    individually, then combined into a final summary.
    """
    if not text.strip():
        raise ValueError("Cannot summarize empty text.")

    summarizer = _get_summarizer()
    chunks = _chunk_text(text, MAX_CHUNK_CHARS)

    chunk_summaries = []
    for chunk in chunks:
        # Skip degenerate tiny chunks that can trip up the model.
        if len(chunk.split()) < 5:
            continue
        out = summarizer(
            chunk, max_length=max_length, min_length=min_length, do_sample=False
        )
        chunk_summaries.append(out[0]["summary_text"].strip())

    combined = " ".join(chunk_summaries)

    # If we had multiple chunks, do one more summarization pass over the
    # combined chunk summaries to produce a single coherent summary.
    if len(chunk_summaries) > 1:
        final = summarizer(
            combined, max_length=max_length, min_length=min_length, do_sample=False
        )[0]["summary_text"].strip()
    else:
        final = combined

    return {
        "summary": final,
        "n_chunks_processed": len(chunk_summaries),
    }


if __name__ == "__main__":
    sample = (
        "The quarterly report shows revenue increased by 12 percent "
        "compared to the previous quarter, driven primarily by growth in "
        "the enterprise segment. Operating costs rose slightly due to "
        "increased hiring in engineering and customer support. The "
        "company expects continued growth next quarter as new product "
        "features roll out to existing customers."
    )
    print(summarize_document(sample))
