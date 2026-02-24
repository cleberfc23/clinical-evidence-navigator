# Clinical Evidence Navigator 🩺📚
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

Clinical Evidence Navigator is an evidence grounded Retrieval Augmented Generation (RAG) system for querying public medical guidelines and returning citation backed answers.

It is built with **LangChain** and **LangGraph** to orchestrate retrieval and response generation workflows, with a strong focus on transparency and hallucination mitigation.

> ⚠️ Disclaimer: This project is for engineering and research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 🎯 Project Goal

Clinical guidelines and medical documents are often lengthy, complex, and difficult to navigate.
This project aims to build a lightweight, accessible, and transparent system that:

- Answers questions **strictly based on retrieved evidence**
- Provides **explicit page-level citations** for each statement
- Restricts LLM responses through controlled context injection
- Reduces hallucinations through prioritized retrieval generation
- Encourages responsible use of AI in healthcare contexts
- Responses are in plain language and do not exceed 300 characters.

---

## 🧩 Tech Stack (v1.0)

- **Python**
- **LangChain** (RAG components, prompts, retrieval)
- **LangGraph** (agent style orchestration and control flow)
- **Chroma** (local vector database)
- **Streamlit** (interactive demo interface)
- **HuggingFace Embeddings** - BAAI/bge-small-en-v1.5
- **Google Gemini 2.5 Flash API**

---

## 🧠 How It Works

The system follows a modular RAG pipeline:

1. Upload a clinical PDF document  
2. Split document into semantic chunks  
3. Generate embeddings for each chunk  
4. Store embeddings in a Chroma vector index  
5. Retrieve top-k relevant chunks  
6. Inject retrieved context into a constrained prompt  
7. Generate a grounded answer with page-level citations  

The model is instructed to respond simply, strictly from retrieved sources.

---

## 🏗 Architecture Overview

app/

├── ingestion.py     
├── generator.py     
├── ui.py            

PS: The other files are for testing purposes.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/clinical-evidence-navigator.git
cd clinical-evidence-navigator
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then open `.env` and add your Gemini API key:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_api_key_here
MODEL_GEMINI_FLASH=gemini-2.5-flash

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 5. Run the application

```bash
streamlit run app/ui.py
```

The app will open in your browser.

---


## 🚧 Project Status

This repository is in MVP stage and under active development.

Planned next steps:
- Vector index caching to avoid rebuilding per query
- Persistent vector database support
- Multi-document ingestion
- URL-based document ingestion
- Automated grounding evaluation metrics
- Backend API version (FastAPI)
- Improved chunking for multi-column PDFs
- Examples of questions

---

## 👤 Author

Cleber F. Carvalho  

---