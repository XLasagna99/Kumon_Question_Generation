import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from vector_store import *

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
    chunks = text_splitter.create_documents([document])
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadata = [{"topic": "football"} for _ in chunks]

    return chunks, ids, metadata

def vector_store_ingestion(topic, chunks, ids, metadata, vector_store_name):
    """Ingest documents into Vector Store."""
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model=str(os.getenv("TEXT_EMBEDDING_MODEL")),
        api_key=str(os.getenv("API_KEY"))
    )
    
    # Check that vector store related to topic exists
    vector_store = get_chroma_client()
    topic_in_registry = topic_exists(topic)
    if topic_in_registry:
        print(f"Topic '{topic}' already exists in registry. Using existing collection.")
        relevant_collection = get_collection(topic_in_registry)
        
    else:
        print(f"Topic '{topic}' not found in registry. Creating new collection.")
        relevant_collection = get_or_create_collection(vector_store_name)
        register_topic(topic, vector_store_name, len(chunks))
    # Add Documents to Vector Store
    relevant_collection.add_documents(
        chunks, 
        ids=ids, 
        metadatas=metadata
    )
    return relevant_collection