# Clinical Evidence Navigator 
![Status](https://img.shields.io/badge/status-active%20development-yellow?style=for-the-badge)
[![Live Demo](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge&logo=streamlit)](https://clinical-evidence-navigator.streamlit.app/)
Clinical Evidence Navigator is a domain-focused Retrieval-Augmented Generation (RAG) system for querying clinical guidelines and generating citation-backed answers.

Designed to demonstrate transparent retrieval workflows, grounded LLM responses, and measurable AI system performance.

> ⚠️ Engineering and research project — not medical advice.

---

## Key Features

- Evidence-grounded clinical question answering  
- Citation-backed responses from medical guidelines  
- Controlled internal document ingestion (no file uploads)  
- Runtime performance monitoring and latency tracking  
- Deployed interactive Streamlit application  

---

## Tech Stack

Python · LangChain · LangGraph · ChromaDB  
HuggingFace Embeddings (bge-small-en-v1.5)
Google Gemini 2.5 Flash · Streamlit

---

## System Metrics

The system tracks execution performance across:

- End-to-end latency  
- Indexing time  
- Retrieval latency  
- LLM inference latency  

Initial optimization reduced end-to-end latency by **~9%** through controlled ingestion redesign.

---

## Run Locally

```bash
git clone https://github.com/your-username/clinical-evidence-navigator.git
cd clinical-evidence-navigator
pip install -r requirements.txt
streamlit run app/ui.py
```

Create .env:
```bash
GEMINI_API_KEY=your_api_key
MODEL_GEMINI_FLASH=gemini-2.5-flash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```
---

## Author 
Cleber F. Carvalho
