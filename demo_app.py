"""
demo_app.py
-----------
Interactive Streamlit demo for the document processing pipeline.
Upload a PDF, see it extracted, classified, and summarized, then ask
questions about it - all running locally, no API key needed.

Run with:
    streamlit run demo_app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import process_document
from src.qa import answer_question

st.set_page_config(page_title="Intelligent Document Processing", page_icon="📄")

st.title("📄 Intelligent Document Processing")
st.write(
    "Upload a PDF and this will extract its text (with automatic OCR for "
    "scanned pages), classify what kind of document it is, and generate a "
    "summary. You can then ask questions about it. Everything runs "
    "locally using open-source models - no API key needed."
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getvalue())

        with st.spinner("Extracting, classifying, and summarizing..."):
            result = process_document(tmp_path)

    if result.get("warning"):
        st.warning(result["warning"])
    else:
        st.subheader("Document type")
        cls = result["classification"]
        st.write(f"**{cls['predicted_label']}** (confidence: {cls['confidence']:.0%})")
        st.bar_chart(cls["all_scores"])

        st.subheader("Summary")
        st.write(result["summary"])

        with st.expander("View extracted text"):
            st.text(result["text"])

        st.session_state["document_text"] = result["text"]

    st.caption(
        f"{result['n_pages']} page(s) processed"
        + (" (OCR was used for scanned pages)" if result["used_ocr"] else "")
    )

if "document_text" in st.session_state:
    st.subheader("Ask a question about this document")
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = answer_question(st.session_state["document_text"], question)
        st.write(f"**Answer:** {answer['answer']}")
        st.caption(f"Confidence: {answer['confidence']:.0%}")
