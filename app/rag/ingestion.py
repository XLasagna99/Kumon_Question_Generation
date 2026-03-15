from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.vector_store import (
    get_collection,
    get_or_create_collection,
    register_topic,
    topic_exists,
)


def rag_document_ingestion(
    textfile: str,
    topic: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> Tuple[List[Document], List[str], List[dict]]:
    """Chunk a text file into LangChain documents for RAG ingestion."""
    document_text = Path(textfile).read_text(encoding="utf-8")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.create_documents([document_text])
    ids = [f"chunk_{index}" for index in range(len(chunks))]
    metadata = [{"topic": topic} for _ in chunks]

    for chunk, chunk_metadata in zip(chunks, metadata):
        chunk.metadata.update(chunk_metadata)

    return chunks, ids, metadata


def vector_store_ingestion(topic, chunks, ids, metadata, vector_store_name):
    """Ingest chunked documents into the topic collection and registry."""
    topic_in_registry = topic_exists(topic)

    if topic_in_registry:
        print(f"Topic '{topic}' already exists in registry. Using existing collection.")
        relevant_collection = get_collection(topic_in_registry)
    else:
        print(f"Topic '{topic}' not found in registry. Creating new collection.")
        relevant_collection = get_or_create_collection(vector_store_name)
        register_topic(topic, vector_store_name, len(chunks))

    documents = _coerce_documents(chunks)
    metadatas = _coerce_metadatas(chunks, metadata)

    relevant_collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    return relevant_collection


def _coerce_documents(chunks: Iterable[Document]) -> List[str]:
    """Convert LangChain documents to plain strings for Chroma."""
    return [chunk.page_content if isinstance(chunk, Document) else str(chunk) for chunk in chunks]


def _coerce_metadatas(chunks, metadata):
    combined_metadata = []

    for chunk, item_metadata in zip(chunks, metadata):
        chunk_metadata = chunk.metadata if isinstance(chunk, Document) else {}
        merged = {**chunk_metadata, **item_metadata}

        if not merged:
            merged = {"source": "unknown"}

        combined_metadata.append(merged)

    return combined_metadata
