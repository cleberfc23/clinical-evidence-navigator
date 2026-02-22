import streamlit as st
import tempfile
# from app.ingestion import create_vectorstore_from_pdf
# from app.config import EMBEDDING_MODEL
from ingestion import create_vectorstore_from_pdf


st.set_page_config(
    page_title="Clinical Evidence Navigator",
    layout="wide"
)

st.title("Clinical Evidence Navigator")
st.markdown("Answers to clinical questions based on scientific evidence, driven by Retrieval Augmented Generation (RAG).")
st.warning("""
⚠️ **Warning:**
This tool is intended for engineering and research purposes only.
It does not provide medical advice and should not replace clinical judgment.""")


uploaded_file = st.file_uploader(
    "Upload a clinical PDF document",
    type=["pdf"])

question = st.text_input(
    "Enter you question"
)


if st.button("Ask"):
    if uploaded_file is None:
        st.error("Please, upload a PDF document first!")
    elif not question:
        st.error("Please enter a question!")
    elif uploaded_file.type != "application/pdf":
        st.error("Invalid file type. Please, upload a PDF document!")
    else:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temporary_document_path = tmp.name

            vectorstore = create_vectorstore_from_pdf(temporary_document_path)

            st.success(
                "Vector store created sucessfully!")


st.markdown("---")
st.markdown("© 2026 Cleber F. Carvalho")
