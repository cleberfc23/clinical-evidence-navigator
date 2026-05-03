# Clinical Evidence Navigator

![Status](https://img.shields.io/badge/development-paused-607d8b?style=for-the-badge)

RAG system for querying clinical guidelines with citation-backed, 
hallucination-mitigated responses. Built with LangGraph, ChromaDB, and FastAPI.

> ⚠️ Engineering and research project — not medical advice.

---

## Architecture
```
PDF Guidelines → Ingestion Pipeline → ChromaDB (bge-small-en-v1.5)
                                           ↓
                              LangGraph Inference Engine
                              (retrieval → reranking → generation)
                                           ↓
                              FastAPI serving layer → Streamlit UI
```

---

## Performance

| Metric              | Before     | After   | Improvement |
|---------------------|------------|---------|-------------|
| End-to-end latency  | ~358s      | ~7.1s   | >98%        |
| Retrieval latency   | tracked    |         |             |
| Indexing time       | tracked    |         |             |
| LLM inference       | tracked    |         |             |

---

## Tech Stack

Python · LangChain · LangGraph · ChromaDB · HuggingFace (bge-small-en-v1.5)  
Google Gemini 2.5 Flash · FastAPI · Streamlit · Docker

---

## Quick Start
```bash
git clone https://github.com/cleberfc23/clinical-evidence-navigator
cd clinical-evidence-navigator
docker compose up
```

---

## Author

[Cleber F. Carvalho](https://github.com/cleberfc23)
