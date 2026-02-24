import streamlit as st
import tempfile
from config import GEMINI_API_KEY, MODEL_GEMINI_FLASH
from ingestion import create_vectorstore_from_pdf
from dotenv import load_dotenv
from google import genai
from generator import build_context, build_prompt
MAX_FILE_SIZE_MB = 20


load_dotenv()

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

user_question = st.text_input(
    "Enter your question"
)

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY. Please seit it in your .env file")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

results = st.container()
if st.button("Ask"):
    results.empty()
    with results:
        if uploaded_file is None:
            st.error("Please, upload a PDF document first!")
            st.stop()
        elif uploaded_file.size > MAX_FILE_SIZE_MB*1024*1024:
            st.error("Please upload a PDF smaller than 20MB")
            st.stop()
        elif not user_question.strip():
            st.error("Please enter a question!")
            st.stop()
        elif len(user_question.strip()) < 8:
            st.error("Please enter a more specific question!")
            st.stop()
        elif uploaded_file.type != "application/pdf":
            st.error("Invalid file type. Please, upload a PDF document!")
            st.stop()
        else:
            with st.spinner("Processing document..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    temporary_document_path = tmp.name

                try:
                    vectorstore = create_vectorstore_from_pdf(
                        temporary_document_path)
                    st.success("Vector store created sucessfully!")
                except Exception as e:
                    st.error("Error processing the uploaded PDF.")
                    st.write(str(e))
                    st.stop()

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                retrieved_docs = retriever.invoke(user_question)
                if not retrieved_docs:
                    st.error(
                        "No relevant content found in the document for this question.")
                    st.stop()

            with st.expander("Show retrieved chunks (debug)"):
                for i, document in enumerate(retrieved_docs):
                    page = document.metadata.get("page", "N/A")
                    st.markdown(f"- Chunk {i+1} (Page {page})")
                    st.write(document.page_content)
                    st.markdown("---")

            context, cited_pages = build_context(retrieved_docs)
            prompt = build_prompt(user_question, context)

            with st.spinner("Generating answer..."):
                try:
                    response = client.models.generate_content(
                        model=MODEL_GEMINI_FLASH,
                        contents=prompt
                    )
                    answer_text = getattr(response, "text", None) or ""

                except Exception as e:
                    st.error("Error while generating response from Gemini API.")
                    st.write(str(e))
                    st.stop()

            st.subheader("Answer")
            if answer_text.strip():
                st.markdown(answer_text)
            else:
                st.warning("No answer has returned")

            st.subheader("Citations")
            if cited_pages:
                for p in cited_pages:
                    try:
                        st.markdown(f"- p. {int(p) + 1}")
                    except Exception:
                        st.markdown(f"- p. {p}")
            else:
                st.markdown("- No page citations available.")


st.markdown("---")
st.markdown("© 2026 Cleber F. Carvalho")
