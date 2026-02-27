import streamlit as st
import tempfile
from config import GEMINI_API_KEY, MODEL_GEMINI_FLASH, EMBEDDING_MODEL
from ingestion import create_vectorstore_from_pdf
from dotenv import load_dotenv
from google import genai
from generator import build_context, build_prompt
import time
import json
from datetime import datetime
from pathlib import Path
import uuid
debug_mode = 1
MAX_FILE_SIZE_MB = 20
MAX_REQUESTS = 3

if "request_count" not in st.session_state:
    st.session_state.request_count = 0


def get_secrets():
    load_dotenv()
    model_name = MODEL_GEMINI_FLASH or st.secrets.get("MODEL_GEMINI_FLASH")
    key = GEMINI_API_KEY or st.secrets.get("GEMINI_API_KEY")
    embedding = EMBEDDING_MODEL or st.secrets.get("EMBEDDING_MODEL")
    return model_name, key, embedding


st.set_page_config(
    page_title="Clinical Evidence Navigator",
    layout="wide"
)
st.title("Clinical Evidence Navigator")
st.markdown("Answers to clinical questions based on scientific evidence, driven by Retrieval Augmented Generation (RAG).")

uploaded_file = st.file_uploader(
    "Upload a clinical PDF document",
    type=["pdf"])

user_question = st.text_input(
    "Enter your question"
)

model_gemini_flash, gemini_api_key, embedding_model = get_secrets()

if not gemini_api_key:
    st.error("Missing GEMINI_API_KEY. Please seit it in your .env file")
    st.stop()

client = genai.Client(api_key=gemini_api_key)

results = st.container()
if st.button("Ask"):
    if st.session_state.request_count >= MAX_REQUESTS:
        st.error(f"You can make only {MAX_REQUESTS} requests!")
        st.stop()
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
                pdf_signature = {
                    "name": uploaded_file.name,
                    "size_bytes": uploaded_file.size
                }
                run_id = str(uuid.uuid4())[:8]
                t0 = time.perf_counter()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    temporary_document_path = tmp.name

                try:
                    t_index_start = time.perf_counter()
                    vectorstore = create_vectorstore_from_pdf(
                        temporary_document_path, embedding_model)
                    index_s = round(
                        (time.perf_counter() - t_index_start), 4)
                    st.success("Vector store created sucessfully!")
                except Exception as e:
                    st.error("Error processing the uploaded PDF.")
                    st.write(str(e))
                    st.stop()

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                t_retrieval_start = time.perf_counter()
                retrieved_docs = retriever.invoke(user_question)
                retrieval_s = round(
                    (time.perf_counter() - t_retrieval_start), 4)
                chunks_retrieved = len(retrieved_docs)
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
                    t_llm_start = time.perf_counter()
                    response = client.models.generate_content(
                        model=model_gemini_flash,
                        contents=prompt
                    )
                    llm_s = round(
                        (time.perf_counter() - t_llm_start), 4)
                    answer_text = getattr(response, "text", None) or ""
                    total_s = round((time.perf_counter() - t0), 4)

                except Exception as e:
                    st.error("Error while generating response from Gemini API.")
                    st.write(str(e))
                    st.stop()

            st.subheader("Answer")
            if answer_text.strip():
                st.info(answer_text)
                st.session_state.request_count += 1

                if debug_mode:
                    row1_col1, row1_col2 = st.columns(2)
                    row2_col1, row2_col2 = st.columns(2)
                    row3_col1, row3_col2 = st.columns(2)

                    row1_col1.metric("End-to-end latency", f"{total_s} s")
                    row1_col2.metric("Indexing time", f"{index_s} s")

                    row2_col1.metric("Retrieval latency", f"{retrieval_s} s")
                    row2_col2.metric("LLM latency", f"{llm_s} s")

                    row3_col1.metric("Chunks retrieved", chunks_retrieved)

                    log_payload = {
                        "app_version": "v0.1.0",
                        "run_id": run_id,
                        "timestamp_utc": datetime.utcnow().isoformat(),
                        "pdf_signature": pdf_signature,
                        "query": user_question,
                        "k": 4,
                        "metrics": {
                            "end_to_end_s": total_s,
                            "indexing_s": index_s,
                            "retrieval_s": retrieval_s,
                            "llm_s": llm_s,
                            "chunks_retrieved": chunks_retrieved
                        }
                    }

                    Path("data").mkdir(exist_ok=True)

                    with open("data/metrics.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_payload,
                                ensure_ascii=False) + "\n")

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
st.warning("""
⚠️ **Warning:**
This tool is intended for engineering and research purposes only.
It does not provide medical advice and should not replace clinical judgment.""")
st.markdown("© 2026 Cleber F. Carvalho")
