import json
from pathlib import Path
import datetime


def serialize_retrieved_docs(retrieved_docs,
                             max_chars: int = 500) -> list[dict]:
    artifacts = []
    for doc in retrieved_docs:
        artifacts.append({
            "page": doc.metadata.get("page"),
            "content": doc.page_content.strip()[:max_chars]
        }
        )
    return artifacts


def build_log_payload(
    app_version: str,
    run_id: str,
    doc_id: str,
    query: str,
    retrieval_top_k: int,
    end_to_end_s: float,
    indexing_s: float,
    retrieval_s: float,
    llm_s: float,
    chunks_retrieved: int,
    retrieved_docs,
    answer_text_by_lmm: str,
    cited_pages: list[int],
    max_artifact_chars: int = 500,
) -> dict:
    serialized_chunks = serialize_retrieved_docs(
        retrieved_docs=retrieved_docs,
        max_chars=max_artifact_chars
    )
    return {
        "app_version": app_version,
        "run_id": run_id,
        "pdf_signature": doc_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "k": retrieval_top_k,
        "metrics": {
            "end_to_end_s": end_to_end_s,
            "indexing_s": indexing_s,
            "retrieval_s": retrieval_s,
            "llm_s": llm_s,
            "chunks_retrieved": chunks_retrieved,
        },
        "artifacs": {
            "answer_text": answer_text_by_lmm,
            "cited_pages": cited_pages,
            "retrieved_chunks": serialized_chunks
        }

    }


def write_log(log_payload):
    data_dir = Path("data")
    Path("data").mkdir(exist_ok=True)
    log_file = data_dir / "metrics.jsonl"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_payload,
                           ensure_ascii=False) + "\n")
