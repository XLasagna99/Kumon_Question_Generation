import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# load in the .env variables
load_dotenv()

def rag_document_ingestion(textfile):
    """Chunk documents from text file for RAG ingestion."""
    with open(textfile) as f:
        document = f.read()
    
    # Initialize Text Splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Create Documents (Chunks) From File
    texts = text_splitter.create_documents([document])
    return texts


def vector_store_name_detection(topic):
    """Generate Vector Store name based on topic."""
    vector_store_name = topic.lower().replace(" ", "_") + "_knowledge_base"
    return vector_store_name


def vector_store_ingestion(texts, vector_store_name):
    """Ingest documents into Vector Store."""
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL")
    )

    # Initialize ChromaDB as Vector Store
    vector_store = Chroma(
        collection_name=vector_store_name,
        embedding_function=embeddings
    )

    # Add Documents to Vector Store
    vector_store.add_documents(texts)
    return vector_store