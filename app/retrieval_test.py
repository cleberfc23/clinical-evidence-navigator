from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def test_retrieval(query: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL
    )

    vector_db = Chroma(
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_DIR, 
        embedding_function = embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs = {"k": k})
    docs = retriever.invoke(query)

    print(f"\n\tQuery: {query}")
    print(f"\tRetrieved: {len(docs)} chunks\n")

    for i, doc in enumerate(docs):
        print(f"Chunk [{i+1}]")
        print(f"Page: {doc.metadata.get('page_number')}")
        print(doc.page_content[:500])
        print("\n")
    

if __name__ == "__main__":
    query_test = "What are the diagnostic criteria for type 2 diabetes"
    query_test = "What are the recommended HbA1c targets for nonpregnant adults?"
    test_retrieval(query_test)