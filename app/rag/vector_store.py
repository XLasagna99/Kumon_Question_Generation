# Initialize Chroma Persistent Vector Stores
import hashlib
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Configuration
# =====================================================

CHROMA_PATH = os.getenv("CHROMA_PATH")
EMBED_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

TOPIC_REGISTRY_COLLECTION = "topic_registry"


# =====================================================
# Internal singletons
# =====================================================

_client: Optional[chromadb.ClientAPI] = None
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
        if not CHROMA_PATH:
            raise ValueError("CHROMA_PATH is not set.")

        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

    return _client


def get_embedding_function():
    """
    Returns a singleton embedding function.
    """
    global _embedding_function

    if _embedding_function is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI API key is not set.")
        if not EMBED_MODEL:
            raise ValueError("TEXT_EMBEDDING_MODEL is not set.")

        _embedding_function = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBED_MODEL,
        )

    return _embedding_function


# =====================================================
# Collection helpers
# =====================================================

def get_or_create_collection(name: str, metadata: Optional[dict] = None):
    client = get_chroma_client()
    embedding_function = get_embedding_function()

    kwargs = {
        "name": name,
        "embedding_function": embedding_function,
    }

    if metadata:
        kwargs["metadata"] = metadata

    return client.get_or_create_collection(**kwargs)


def get_collection(name: str):
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
        metadata={
            "purpose": "topic lookup registry",
            "hnsw:space": "cosine",
        },
    )


def reset_topic_registry():
    """
    Delete and recreate the topic registry collection.

    Use this after changing registry distance settings so Chroma rebuilds the
    collection with the new metadata.
    """
    client = get_chroma_client()

    try:
        client.delete_collection(TOPIC_REGISTRY_COLLECTION)
    except Exception:
        pass

    return get_topic_registry()


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
            "topic": topic,
            "collection_name": collection_name,
            "source_count": source_count,
        }],
    )


def topic_exists(
    topic: str,
    similarity_threshold: float = 0.6,
    n_results: int = 3,
    return_matches: bool = False,
):
    """
    Check whether a topic or a semantically similar topic exists in the registry.

    By default, returns the best matching collection name if it meets the
    similarity threshold. If ``return_matches`` is True, returns a list of
    matching topics with similarity scores instead.
    """
    registry = get_topic_registry()

    if registry.count() == 0:
        return [] if return_matches else None

    result = registry.query(
        query_texts=[topic],
        n_results=n_results,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not documents:
        return [] if return_matches else None

    matches: List[dict] = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = 1 - distance
        if similarity < similarity_threshold:
            continue

        metadata = metadata or {}
        matches.append(
            {
                "topic": metadata.get("topic", document),
                "collection_name": metadata.get("collection_name"),
                "similarity": similarity,
            }
        )

    if return_matches:
        return matches

    if matches:
        return matches[0]["collection_name"]

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
