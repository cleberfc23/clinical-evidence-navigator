import streamlit as st
from core.settings import DEFAULT_DOC, get_secrets, validate_runtime_config, DEBUG_MODE, MAX_REQUESTS, RETRIEVAL_TOP_K
import time
import uuid
from app.generator import generate_answer, build_client
from core.observability import write_log, build_log_payload
from ingestion.pipeline import load_vectorstore, create_vectorstore

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

run_time_config_dict = get_secrets()
missing_fields = validate_runtime_config(run_time_config_dict)
model_gemini_flash = run_time_config_dict["model_name"]
gemini_api_key = run_time_config_dict["api_key"]
embedding_model = run_time_config_dict["embedding_model"]

if missing_fields:
    st.error(
        "Missing required configuration: " + ", ".join(missing_fields)
    )
    st.stop()


st.set_page_config(
    page_title="Clinical Evidence Navigator",
    layout="wide"
)
st.title("Clinical Evidence Navigator")
st.markdown("Answers to clinical questions based on scientific evidence, driven by Retrieval Augmented Generation (RAG).")
st.caption("Fields covered: \n - Diabetes (Standards of Care 2026)")

user_question = st.text_input(
    "Enter your question"
)
client = build_client(gemini_api_key)
results = st.container()
if st.button("Ask"):
    if st.session_state.request_count >= MAX_REQUESTS:
        st.error(f"You can make only {MAX_REQUESTS} requests!")
        st.stop()
    results.empty()

    with results:
        if not user_question.strip():
            st.error("Please enter a question!")
            st.stop()
        elif len(user_question.strip()) < 8:
            st.error("Please enter a more specific question!")
            st.stop()
        else:
            run_id = str(uuid.uuid4())[:8]
            t0 = time.perf_counter()

            with st.spinner("Preparing document index..."):
                try:
                    t_index_start = time.perf_counter()
                    vectorstore = load_vectorstore(embedding_model)
                    # vectorstore = create_vectorstore(embedding_model)
                    metric_index_s = round(
                        time.perf_counter() - t_index_start, 4)
                    st.success("Vector store created successfully!")
                except Exception as e:
                    st.error("Error while preparing the document index.")
                    st.write(str(e))
                    st.stop()

            with st.spinner("Retrieving relevant evidence..."):
                try:
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": RETRIEVAL_TOP_K})

                    t_retrieval_start = time.perf_counter()
                    retrieved_docs = retriever.invoke(user_question)
                    metric_retrieval_s = round(
                        time.perf_counter() - t_retrieval_start, 4)
                    metric_chunks_retrieved = len(retrieved_docs)

                    if not retrieved_docs:
                        st.error("No relevant content found for this question.")
                        st.stop()

                except Exception as e:
                    st.error("Error while retrieving relevant content.")
                    st.write(str(e))
                    st.stop()

            with st.expander("Show retrieved chunks (debug)"):
                for i, document in enumerate(retrieved_docs):
                    page = document.metadata.get("page", "N/A")
                    st.markdown(f"- Chunk {i+1} (Page {page})")
                    st.write(document.page_content)
                    st.markdown("---")

            with st.spinner("Generating answer..."):
                try:
                    t_llm_start = time.perf_counter()
                    generation_result = generate_answer(
                        client=client,
                        model_name=model_gemini_flash,
                        user_question=user_question,
                        retrieved_docs=retrieved_docs,
                    )
                    metric_llm_s = round(
                        (time.perf_counter() - t_llm_start), 4)
                    answer_text = generation_result["answer_text"]
                    cited_pages = generation_result["cited_pages"]
                    metric_total_s = round((time.perf_counter() - t0), 4)

                except Exception as e:
                    st.error("Error while generating response from Gemini API.")
                    st.write(str(e))
                    st.stop()

            st.subheader("Answer")
            if answer_text.strip():
                st.info(answer_text)
                st.session_state.request_count += 1

                if DEBUG_MODE:
                    row1_col1, row1_col2 = st.columns(2)
                    row2_col1, row2_col2 = st.columns(2)
                    row3_col1, row3_col2 = st.columns(2)

                    row1_col1.metric("End-to-end latency",
                                     f"{metric_total_s} s")
                    row1_col2.metric("Indexing time", f"{metric_index_s} s")

                    row2_col1.metric("Retrieval latency",
                                     f"{metric_retrieval_s} s")
                    row2_col2.metric("LLM latency", f"{metric_llm_s} s")

                    row3_col1.metric("Chunks retrieved",
                                     metric_chunks_retrieved)

                    log_payload = build_log_payload(
                        app_version="v0.1.0",
                        run_id=run_id,
                        doc_id=DEFAULT_DOC["doc_id"],
                        query=user_question,
                        retrieval_top_k=RETRIEVAL_TOP_K,
                        end_to_end_s=metric_total_s,
                        indexing_s=metric_index_s,
                        retrieval_s=metric_retrieval_s,
                        llm_s=metric_llm_s,
                        chunks_retrieved=metric_chunks_retrieved,
                        retrieved_docs=retrieved_docs,
                        answer_text_by_lmm=answer_text,
                        cited_pages=cited_pages
                    )
                    write_log(log_payload)

            else:
                st.warning("No answer was returned")

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
