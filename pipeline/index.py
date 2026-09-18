"""
Embedding + indexing.

Embeddings: sentence-transformers all-MiniLM-L6-v2, run locally via
langchain_huggingface -- free, no API key, no rate limit, 384-dim.
Tradeoff worth knowing: it's a smaller/older model than OpenAI's

Vector store: Qdrant, so metadata filtering (by content_type, source,
table_id) is available at query time
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from pipeline.chunker import Chunk

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed for all-MiniLM-L6-v2
HASH_INDEX_PATH = Path(__file__).resolve().parents[1] / ".rag_hash_index.json"


def session_collection_name(session_id: str) -> str:
    """Return the Qdrant collection name for one logical session."""
    return f"docs_{session_id}"


def _load_hash_index() -> dict[str, dict[str, str]]:
    if not HASH_INDEX_PATH.exists():
        return {}
    try:
        with HASH_INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_hash_index(index: dict[str, dict[str, str]]) -> None:
    with HASH_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def get_hash_record(file_hash: str) -> dict[str, str] | None:
    """Return the previous session record for this file hash, if any."""
    index = _load_hash_index()
    record = index.get(file_hash)
    if not record:
        return None
    return record if isinstance(record, dict) else None


def register_hash(file_hash: str, session_id: str, collection_name: str) -> None:
    """Persist the fact that this exact file hash is already available in a session collection."""
    index = _load_hash_index()
    index[file_hash] = {
        "session_id": session_id,
        "collection_name": collection_name,
    }
    _save_hash_index(index)


def clear_stale_hash_record(file_hash: str) -> None:
    """Remove a hash mapping when the referenced collection no longer exists."""
    index = _load_hash_index()
    index.pop(file_hash, None)
    _save_hash_index(index)


def collection_has_file_hash(client, collection_name: str, file_hash: str) -> bool:
    """True when this exact file content has already been indexed in the session collection."""
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    hits, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="metadata.file_hash", match=MatchValue(value=file_hash))]
        ),
        limit=1,
        with_payload=True,
    )
    return bool(hits)


def copy_hash_chunks_to_collection(source_collection_name: str, target_collection_name: str, file_hash: str,
                                  qdrant_url: str = "http://localhost:6333") -> int:
    """Copy the existing vector chunks for a file hash from a prior collection into the current session collection."""
    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(source_collection_name):
        clear_stale_hash_record(file_hash)
        return 0
    if not client.collection_exists(target_collection_name):
        client.create_collection(
            collection_name=target_collection_name,
            vectors_config={"size": EMBEDDING_DIM, "distance": "Cosine"},
        )

    hits, _ = client.scroll(
        collection_name=source_collection_name,
        scroll_filter={"must": [{"key": "metadata.file_hash", "match": {"value": file_hash}}]},
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    if not hits:
        return 0

    docs = []
    for point in hits:
        payload = point.payload or {}
        metadata = payload.get("metadata", {}) or {}
        docs.append(Document(
            page_content=payload.get("page_content", "") or "",
            metadata={
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page"),
                "content_type": metadata.get("content_type", "text"),
                "table_id": metadata.get("table_id"),
                "chunk_index": metadata.get("chunk_index"),
                "file_hash": file_hash,
                **{k: v for k, v in metadata.items() if k not in {"source", "page", "content_type", "table_id", "chunk_index"}},
            },
        ))

    if not docs:
        return 0

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=target_collection_name,
        embedding=get_embeddings(),
    )
    vectorstore.add_documents(documents=docs)
    return len(docs)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(chunks: list[Chunk], collection_name: str = "docs",
                       qdrant_url: str = "http://localhost:6333",
                       file_hash: str | None = None):
    """file_hash is stamped onto every chunk's metadata so pipeline/sync.py
    can later detect whether this source file has changed since it was
    last indexed, without re-reading/re-chunking/re-embedding it."""
    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    embeddings = get_embeddings()
    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    documents = [
        Document(
            page_content=c.text,
            metadata={
                "source": c.source,
                "page": c.page,
                "content_type": c.content_type,
                "table_id": c.table_id,
                "chunk_index": c.chunk_index,
                "file_hash": file_hash,
                **c.metadata,
            },
        )
        for c in chunks
    ]
    ids = [str(uuid.uuid4()) for _ in documents]

    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings,
    )
    if documents:
        vectorstore.add_documents(documents=documents, ids=ids)
    return vectorstore


def reset_collection(collection_name: str = "docs",
                    qdrant_url: str = "http://localhost:6333"):
    """Clear the collection so the next upload is the only source of truth.
    This prevents stale chunks from previous uploads from still being retrieved."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    return client


def get_vectorstore(collection_name: str = "docs",
                     qdrant_url: str = "http://localhost:6333"):
    # """Connects to an existing (or empty) collection without adding
    # anything -- used by the sync-aware ingestion path, which adds
    # chunks file-by-file after deciding what needs (re-)embedding."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    embeddings = get_embeddings()
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection_name):
        print(f"Creating new Qdrant collection '{collection_name}' for the uploaded document.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Using existing Qdrant collection '{collection_name}' for the uploaded document.")
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings), client


def build_bm25_retriever(chunks: list[Chunk]):
    """Sparse retriever for hybrid search -- catches exact keyword/ID
    matches (e.g. a specific product code in a table) that dense
    retrieval alone can miss."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    documents = [
        Document(page_content=c.text, metadata={
            "source": c.source, "page": c.page, "content_type": c.content_type,
            "table_id": c.table_id,
        })
        for c in chunks
    ]
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 10
    return retriever


def build_hybrid_retriever(vectorstore, bm25_retriever, dense_k: int = 10, weights=(0.4, 0.6)):
    from langchain_classic.retrievers import EnsembleRetriever

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": dense_k})
    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=list(weights),
    )