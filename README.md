# Clinical Evidence Navigator 🩺📚

Clinical Evidence Navigator is an evidence grounded Retrieval Augmented Generation (RAG) system for querying public medical guidelines and returning citation backed answers.

It is built with **LangChain** and **LangGraph** to orchestrate retrieval and response generation workflows, with a strong focus on transparency and hallucination mitigation.

> ⚠️ Disclaimer: This project is for research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 🎯 Project Goal

Medical guidelines are often long, complex, and difficult to navigate.  
This project aims to build a lightweight and transparent system that:

- Answers questions **strictly based on retrieved evidence**
- Provides **explicit citations** for every claim
- Detects when a question is **out of scope** of the available documents
- Encourages **responsible AI usage** in healthcare contexts

---

## 🧩 Tech Stack (v1.0)

- **Python**
- **LangChain** (RAG components, prompts, retrieval)
- **LangGraph** (agent style orchestration and control flow)
- **Chroma** (local vector database)
- **Streamlit** (demo interface — upcoming)

---

## 🚧 Project Status

This repository is under active development.

Planned next steps:
- Document ingestion pipeline (PDF/MD/HTML)
- Vector indexing and retrieval setup
- Streamlit demo app
- Basic evaluation (citation coverage / grounding checks)

---
