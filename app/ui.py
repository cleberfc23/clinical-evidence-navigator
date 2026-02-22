import streamlit as st

st.set_page_config(
    page_title="Clinical Evidence Navigator",
    page_icon="🩺",
    layout="wide"
)

st.title("Clinical Evidence Navigator")
st.markdown("Answers to clinical questions based on scientific evidence, driven by Retrieval Augmented Generation (RAG).")
st.warning("""
⚠️ **Warning:**
This tool is intended for research and engineering purposes only.
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
    elif uploaded_file.type !="application/pdf":
        st.error("Invalid file type. Please, upload a PDF document!")
    else:
        st.sucess(
            "MVP structure working.\n Next step: connect to retrieval pipeline")


st.markdown("---")
st.markdown("© 2026 Cleber F. Carvalho")