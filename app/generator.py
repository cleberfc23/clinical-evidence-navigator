from typing import List, Tuple
from langchain_core.documents import Document


def build_context(docs: List[Document], max_chars: int = 12000) -> Tuple[str, List[int]]:
    """
    """
    parts = []
    cited_pages = []
    total = 0
    
    for d in docs:
        page = d.metadata.get("page", None)
        if page is not None:
            cited_pages.append(page)

        snippet = d.page_content.strip()
        block = f"[SOURCE: page={page}]\n{snippet}"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    seen = set()
    cited_pages_unique = []
    for p in cited_pages:
        if p not in seen:
            cited_pages_unique.append(p)
            seen.add(p)

    return "\n---\n".join(parts), cited_pages_unique


def build_prompt(question: str, context: str) -> str:
    """
    Prompt engineered for grounded QA + citations by page
    """
    return f"""
You are a clinical evidence assistant. Answer strictly using the provided SOURCES. 
If the SOURCES do not contain enough information, say you don't know.

Rules:
- Be concise and clinically precise.
- Do not invent facts or recomendations.
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
