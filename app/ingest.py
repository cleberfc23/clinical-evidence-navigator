from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import PDF_PATH_DIABETES, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODE


def ingest():
    if not PDF_PATH_DIABETES:
        raise ValueError("PDF_PATH is not defined.")
    
    if Path(CHROMA_DIR).exists():
        print("Chroma directory already exists. Skip!")
        return 
    
    print(" === Ingestion START! === ")

    loader = PyPDFLoader(PDF_PATH_DIABETES)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150
    )

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME
    )

    vectordb = Chroma(
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_DIR,
        embedding_function = embeddings
    )
    
    vectordb.add_documents(chunks)
    vectordb.persist()
    print(" === Ingestion Complete === ")



if __name__ == "__main__":   
    ingest()


