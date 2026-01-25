from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def load_vector_store(persist_dir: str):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
