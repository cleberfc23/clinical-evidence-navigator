from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import tempfile
from config import DEFAULT_DOC, MAX_FILE_SIZE_MB, CHUNK_SIZE, CHUNK_OVERLAP


def download_to_tempfile(url: str, max_mb: int = 20) -> str:
    max_bytes = max_mb * 1024 * 1024
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    content_type = (r.headers.get("Content-Type") or "").lower()
    pdf_bytes = r.content

    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF too large: {len(pdf_bytes)} bytes")

    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF (Content-Type: {content_type})")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


def load_documents(url_path: str):
    loader = PyPDFLoader(url_path)
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_embeddings(model: str):
    return HuggingFaceEmbeddings(model_name=model)


def build_vectorstore(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )


def create_vectorstore(embedding_model: str):
    url_path = download_to_tempfile(
        url=DEFAULT_DOC["url"],
        max_mb=MAX_FILE_SIZE_MB
    )

    documents_loader = load_documents(url_path)

    chunks = split_documents(
        documents=documents_loader,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    embeddings = build_embeddings(embedding_model)
    vectorstore = build_vectorstore(chunks, embeddings)

    return vectorstore