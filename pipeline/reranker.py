"""
Cross-encoder reranking. Runs on the small candidate set returned by hybrid search
"""
from __future__ import annotations

import functools


# def build_reranker(top_n: int = 5):
#     """Cohere Rerank is used here for simplicity; swap for a local
#     cross-encoder (e.g. sentence-transformers `bge-reranker`) if you
#     need to avoid an external API dependency for the take-home."""
#     from langchain_cohere import CohereRerank
#     return CohereRerank(model="rerank-english-v3.0", top_n=top_n)


@functools.lru_cache(maxsize=1)
def load_cross_encoder():
    """The actual heavy model load, cached separately from top_n so
    that build_reranker(top_n=5) and build_reranker(top_n=3)
    reuse the SAME loaded model instead of each triggering a fresh
    disk load -- top_n is just a slicing parameter on the wrapper
    below, not something that should invalidate the cache."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-base")


def build_reranker(top_n: int = 5):
    """No-external-API alternative using a local cross-encoder model.
    The model itself loads once per process (see load_cross_encoder
    above) -- this function is now cheap to call repeatedly, which
    matters for Streamlit's rerun-the-whole-script execution model."""
    from langchain_core.documents import Document

    model = load_cross_encoder()

    class LocalReranker:
        def compress_documents(self, documents: list[Document], query: str) -> list[Document]:
            if not documents:
                return []
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_n]]

    return LocalReranker()