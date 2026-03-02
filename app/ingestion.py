from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import tempfile
from config import DEFAULT_DOC, MAX_FILE_SIZE_MB


def download_pdf_to_tempfile(url: str, max_mb: int = 20) -> str:
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


def create_vectorstore_from_pdf(embedding_model):
    temporary_document_path = download_pdf_to_tempfile(
        DEFAULT_DOC["url"], MAX_FILE_SIZE_MB)
    loader = PyPDFLoader(temporary_document_path)
    document_pdf = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(document_pdf)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore
