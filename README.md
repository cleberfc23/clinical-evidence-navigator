# Clinical Evidence Navigator 
![Status](https://img.shields.io/badge/status-active%20development-yellow?style=for-the-badge)


Clinical Evidence Navigator is a domain-focused Retrieval-Augmented Generation (RAG) system for querying clinical guidelines and generating citation-backed answers.

Designed to demonstrate transparent retrieval workflows, grounded LLM responses, and measurable AI system performance.

> ⚠️ Engineering and research project — not medical advice.

---

## Key Features

- Evidence-grounded clinical question answering  
- Citation-backed responses from medical guidelines  
- Modular architecture (ingestion, core inference, serving)  
- API-based access via FastAPI  
- Runtime performance monitoring and latency tracking  


---

## Tech Stack

Python · LangChain · LangGraph · ChromaDB  
HuggingFace Embeddings (bge-small-en-v1.5)  
Google Gemini 2.5 Flash · FastAPI · Streamlit · Docker  

---

## System Metrics

The system tracks execution performance across:

- End-to-end latency  
- Indexing time  
- Retrieval latency  
- LLM inference latency  

Optimized end-to-end latency from ~358s to ~7.1s (>98% improvement).

---



## Author 
Cleber F. Carvalho
