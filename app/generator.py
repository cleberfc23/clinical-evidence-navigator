from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from google import genai


def build_client(key):
    return genai.Client(api_key=key)


def build_context(
    docs: List[Document],
    max_chars: int = 12000
) -> Tuple[str, List[int]]:
    parts = []
    cited_pages = []
    total_chars = 0

    for doc in docs:
        page = doc.metadata.get("page")
        snippet = doc.page_content.strip()

        if not snippet:
            continue

        if page is not None:
            block = f"[SOURCE: page={page}]\n{snippet}"
        else:
            block = "[SOURCE]\n" + snippet

        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += len(block)

        if page is not None and page not in cited_pages:
            cited_pages.append(page)

    context = "\n---\n".join(parts)
    return context, cited_pages


def build_prompt(question: str, context: str) -> str:
    """
    Prompt engineered for grounded QA + citations by page
    """
    return f"""
            You are a clinical evidence assistant. Answer strictly using the provided SOURCES.
            If the SOURCES do not contain enough information, say you don't know.

            Rules:
            - Be concise and clinically precise.
            - Do not invent facts or recommendations.
            - Always include citations as (p. X) for every key claim, where X is the page number from sources
            - Use a very simple english
            - You answer MUST NOT have more than 300 characters
            - Do not repeat the question. Provide the answer only.

            QUESTION:
            {question}

            SOURCES:
            {context}

            Now write the answer:
            """.strip()


def generate_answer(
        client,
        model_name: str,
        user_question: str,
        retrieved_docs: List[Document],
        max_context_chars: int = 12000
) -> Dict[str, Any]:
    context, cited_pages = build_context(
        docs=retrieved_docs,
        max_chars=max_context_chars
    )
    prompt = build_prompt(question=user_question, context=context)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    answer_text = getattr(response, "text", None) or ""
    return {
        "answer_text": answer_text.strip(),
        "cited_pages": cited_pages,
        "context": context,
        "prompt": prompt
    }
