from langchain_chroma import Chroma

PERSIST_DIRECTORY = "data/processed/chroma"


def load_vectorstore(embedding_model):
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
    )