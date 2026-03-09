# Initialize Chroma Persistent Vector Stores
import os
import hashlib
from typing import Optional, Dict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# =====================================================
# Configuration
# =====================================================

CHROMA_PATH = os.getenv("CHROMA_PATH")
EMBED_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")

TOPIC_REGISTRY_COLLECTION = "topic_registry"


# =====================================================
# Internal singletons
# =====================================================

_client = None
_embedding_function = None


# =====================================================
# Core initializers
# =====================================================

def get_chroma_client():
    """
    Returns a singleton Chroma persistent client.
    """
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)

    return _client


def get_embedding_function():
    """
    Returns a singleton embedding function.
    """
    global _embedding_function

    if _embedding_function is None:
        _embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

    return _embedding_function


# =====================================================
# Collection helpers
# =====================================================

def get_or_create_collection(name, metadata = None):
    """
    Retrieve or create a Chroma collection.
    """
    client = get_chroma_client()
    embedding_function = get_embedding_function()

    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
        metadata=metadata or {}
    )


def get_collection(name):
    """
    Retrieve an existing collection.
    """
    client = get_chroma_client()
    embedding_function = get_embedding_function()

    return client.get_collection(
        name=name,
        embedding_function=embedding_function
    )


# =====================================================
# Topic Registry
# =====================================================

def get_topic_registry():
    """
    Returns the topic registry collection.
    """
    return get_or_create_collection(
        TOPIC_REGISTRY_COLLECTION,
        metadata={"purpose": "topic lookup registry"}
    )


def register_topic(topic, collection_name, source_count):
    """
    Register a topic in the registry.
    """
    registry = get_topic_registry()

    topic_id = _topic_hash(topic)

    registry.upsert(
        ids=[topic_id],
        documents=[topic],
        metadatas=[{
            "collection_name": collection_name,
            "source_count": source_count
        }]
    )


def topic_exists(topic, similarity_threshold= 0.82):
    """
    Checks if a topic already exists in the registry.
    Returns collection name if found.
    """
    registry = get_topic_registry()

    if registry.count() == 0:
        return None

    result = registry.query(
        query_texts=[topic],
        n_results=1
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not documents:
        return None

    similarity = 1 - distances[0]

    if similarity >= similarity_threshold:
        return metadatas[0]["collection_name"]

    return None


# =====================================================
# Topic collections
# =====================================================

def topic_collection_name(topic: str) -> str:
    """
    Converts topic string into a standardized collection name.
    """
    topic_slug = topic.lower().replace(" ", "_")
    return f"topic_{topic_slug}"


def get_topic_collection(topic: str):
    """
    Returns or creates a topic collection.
    """
    collection_name = topic_collection_name(topic)

    return get_or_create_collection(
        collection_name,
        metadata={"topic": topic}
    )


# =====================================================
# Utility
# =====================================================

def _topic_hash(topic: str) -> str:
    """
    Deterministic ID for topic registry entries.
    """
    return hashlib.sha1(topic.lower().encode()).hexdigest()