import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )

    # Create Documents (Chunks) From File
    texts = text_splitter.create_documents([document])
    return texts


def vector_store_name_detection(topic):
    """Generate Vector Store name based on topic."""
    vector_store_name = topic.lower().replace(" ", "_") + "_knowledge_base"
    return vector_store_name

def vector_store_creation_based_on_extrapolated_topic(texts):
    """Create a Vector Store and persist it based on the determined topic of the read text"""


def vector_store_ingestion(texts, vector_store_name):
    """Ingest documents into Vector Store."""
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model=str(os.getenv("TEXT_EMBEDDING_MODEL")),
        api_key=str(os.getenv("API_KEY"))
    )

    # Initialize ChromaDB as Vector Store
    vector_store = Chroma(
        collection_name=vector_store_name,
        embedding_function=embeddings,
        persist_directory=f"./app/data/seed_context/{vector_store_name}"
    )

    # Add Documents to Vector Store
    vector_store.add_documents(texts)
    return vector_store