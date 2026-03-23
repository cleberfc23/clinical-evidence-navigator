import time

from ingestion.pipeline import load_vectorstore

from app.generator import generate_answer


def answer_question(
    user_question: str,
    embedding_model,
    client,
    model_name: str,
    top_k: int,
):
    if not user_question.strip():
        raise ValueError("Please enter a question!")

    if len(user_question.strip()) < 8:
        raise ValueError("Please enter a more specific question!")

    t0 = time.perf_counter()

    vectorstore = load_vectorstore(embedding_model)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    t_retrieval_start = time.perf_counter()
    retrieved_docs = retriever.invoke(user_question)
    metric_retrieval_s = round(time.perf_counter() - t_retrieval_start, 4)
    metric_chunks_retrieved = len(retrieved_docs)

    if not retrieved_docs:
        raise ValueError("No relevant content found for this question.")

    t_llm_start = time.perf_counter()
    generation_result = generate_answer(
        client=client,
        model_name=model_name,
        user_question=user_question,
        retrieved_docs=retrieved_docs,
    )
    metric_llm_s = round(time.perf_counter() - t_llm_start, 4)
    metric_total_s = round(time.perf_counter() - t0, 4)

    return {
        "answer_text": generation_result["answer_text"],
        "cited_pages": generation_result["cited_pages"],
        "retrieved_docs": retrieved_docs,
        "metrics": {
            "retrieval_s": metric_retrieval_s,
            "llm_s": metric_llm_s,
            "total_s": metric_total_s,
            "chunks_retrieved": metric_chunks_retrieved,
        },
    }