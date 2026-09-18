"""
End-to-end CLI entrypoint.

Usage:
    export OPENAI_API_KEY=...     # for OpenAI generation + vision OCR fallback
    # or: export GEMINI_API_KEY=... and pass model_name="gemini-2.5-flash"

    docker run -p 6333:6333 qdrant/qdrant   # local Qdrant instance

    python main.py ./sample_docs "What was the Q3 revenue by region?"
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from pipeline.ingest import ingest_directory
from pipeline.index import get_vectorstore
from pipeline.index import build_bm25_retriever, build_hybrid_retriever
from pipeline.reranker import build_reranker
from agent.rag_agent import build_agent, ask

COLLECTION_NAME = "docs"
QDRANT_URL = "http://localhost:6333"


def main(doc_dir: str, question: str):
    print("=== Syncing documents into the vector index ===")
    ingest_directory(doc_dir, collection_name=COLLECTION_NAME, qdrant_url=QDRANT_URL)

    print("\n=== Connecting to vector store + building hybrid retriever ===")
    vectorstore, client = get_vectorstore(COLLECTION_NAME, QDRANT_URL)

    # Getting all chunks from the vector store to build BM25 for hybrid search.
    all_points = client.scroll(collection_name=COLLECTION_NAME, limit=10_000, with_payload=True)[0]
    from pipeline.chunker import Chunk
    all_chunks = [
        Chunk(
            text=p.payload.get("page_content", "") or "",
            content_type=p.payload.get("metadata", {}).get("content_type", "text"),
            source=p.payload.get("metadata", {}).get("source", "unknown"),
            page=p.payload.get("metadata", {}).get("page"),
            table_id=p.payload.get("metadata", {}).get("table_id"),
        )
        for p in all_points
    ]
    bm25 = build_bm25_retriever(all_chunks) if all_chunks else None
    hybrid = build_hybrid_retriever(vectorstore, bm25) if bm25 else vectorstore.as_retriever()

    print("=== Loading reranker (local cross-encoder) ===")
    reranker = build_reranker(top_n=5)

    print("=== Building deep agent (OpenAI) ===")
    agent = build_agent(hybrid, reranker=reranker)

    print(f"\n=== Asking: {question} ===")
    # hybrid + reranker are passed explicitly -- ask() retrieves
    # deterministically before invoking the agent, it no longer
    # depends on tools bound at agent-build time for the primary
    # retrieval pass.
    answer = ask(agent, question, hybrid, reranker=reranker)
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <doc_directory> <question>")
        sys.exit(1)
    main(sys.argv[1], " ".join(sys.argv[2:]))