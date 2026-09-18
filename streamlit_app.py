"""
Streamlit UI for testing the pipeline end-to-end: upload files, run
ingestion, inspect what was extracted, and ask questions against the
deep agent.

Run:
    export OPENAI_API_KEY=...   # and/or GROQ_API_KEY=... for the orchestrator model
    docker run -p 6333:6333 qdrant/qdrant
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from ingestion.loaders import load_file, LOADERS
from pipeline.chunker import chunk_all
from pipeline.index import (
    build_bm25_retriever,
    build_hybrid_retriever,
    build_vectorstore,
    collection_has_file_hash,
    copy_hash_chunks_to_collection,
    get_hash_record,
    get_vectorstore,
    register_hash,
    reset_collection,
    session_collection_name,
)
from pipeline.reranker import build_reranker
from pipeline.sync import decide_sync_action, delete_source_chunks, SyncDecision
from agent.rag_agent import build_agent, ask_with_trace

st.set_page_config(page_title="Document RAG Agent", layout="wide")
st.title("📄 Multi-Format Document RAG Agent")
st.caption("PDF (incl. scanned) · PPTX · XLSX/CSV — with table-aware chunking and a deepagents-based Q&A agent")

DEFAULT_QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "openai:gpt-4o-mini"

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    max_chars = st.number_input("Chunk size (characters)", value=500, min_value=100, step=50)
    overlap_chars = st.number_input("Chunk overlap (characters)", value=100, min_value=0, step=25)
    st.markdown("---")
    show_debug = st.checkbox("Show retrieval debug panel", value=True,
                              help="Shows exactly what was retrieved for each question, "
                                   "so you can verify the answer is actually grounded "
                                   "instead of just trusting it.")

# Session state
for key in ["agent", "hybrid_retriever", "reranker", "chunks", "ingest_log"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

current_collection = session_collection_name(st.session_state.session_id)

# Step 1: upload + ingest

st.subheader("1. Upload documents")
st.caption(f"Current session: {st.session_state.session_id[:8]}")
uploaded_files = st.file_uploader(
    "Upload PDF, PPTX, XLSX, XLS, or CSV files (multiple allowed)",
    type=["pdf", "pptx", "xlsx", "xls", "csv"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 3])
with col1:
    run_ingest = st.button("Run ingestion", type="primary", disabled=not uploaded_files)

with col2:
    new_session = st.button("Start fresh session")

if new_session:
    st.session_state.session_id = uuid.uuid4().hex
    current_collection = session_collection_name(st.session_state.session_id)
    reset_collection(current_collection, DEFAULT_QDRANT_URL)
    st.session_state.agent = None
    st.session_state.hybrid_retriever = None
    st.session_state.reranker = None
    st.session_state.chunks = []
    st.session_state.ingest_log = []
    st.success("Started a fresh session. Previous session data is isolated and no longer in scope.")

if run_ingest and uploaded_files:
    st.session_state.agent = None
    st.session_state.hybrid_retriever = None
    st.session_state.reranker = None
    st.session_state.chunks = []
    st.session_state.ingest_log = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    try:
        for f in uploaded_files:
            (tmp_dir / f.name).write_bytes(f.getbuffer())

        progress = st.progress(0.0, text="Starting ingestion...")
        log_lines = []
        all_chunks = []
        files = list(tmp_dir.glob("*"))

        vectorstore, qdrant_client = get_vectorstore(current_collection, DEFAULT_QDRANT_URL)

        for i, path in enumerate(files):
            ext = path.suffix.lower()
            if ext not in LOADERS:
                continue
            progress.progress((i + 1) / len(files), text=f"Checking {path.name}...")

            file_bytes = path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            # Original sync guard: skip unchanged items in this session collection.
            decision, file_hash = decide_sync_action(qdrant_client, current_collection, path)
            if decision == SyncDecision.SKIP_UNCHANGED:
                log_lines.append(f"⏭️ **{path.name}**: unchanged in this session, skipped")
                continue
            if decision == SyncDecision.UPDATED_FILE:
                deleted = delete_source_chunks(qdrant_client, current_collection, str(path))
                log_lines.append(f"🔄 **{path.name}**: content changed, removed {deleted} stale chunk(s)")

            if collection_has_file_hash(qdrant_client, current_collection, file_hash):
                log_lines.append(f"⏭️ **{path.name}**: already indexed in this session, skipped")
                continue

            previous_record = get_hash_record(file_hash)
            if previous_record and previous_record.get("collection_name") != current_collection:
                source_collection = previous_record["collection_name"]
                reused = copy_hash_chunks_to_collection(
                    source_collection,
                    current_collection,
                    file_hash,
                    qdrant_url=DEFAULT_QDRANT_URL,
                )
                if reused:
                    register_hash(file_hash, st.session_state.session_id, current_collection)
                    log_lines.append(
                        f"♻️ **{path.name}**: reused existing indexed content from prior session "
                        f"({source_collection}) without re-embedding"
                    )
                    continue

            progress.progress((i + 1) / len(files), text=f"Processing {path.name}...")
            try:
                blocks = load_file(path)
                chunks = chunk_all(
                    blocks,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
                build_vectorstore(
                    chunks,
                    collection_name=current_collection,
                    qdrant_url=DEFAULT_QDRANT_URL,
                    file_hash=file_hash,
                )
                register_hash(file_hash, st.session_state.session_id, current_collection)
                all_chunks.extend(chunks)
                n_text = sum(1 for b in blocks if b.content_type == "text")
                n_table = sum(1 for b in blocks if b.content_type == "table")
                n_ocr = sum(1 for b in blocks if b.content_type == "image_ocr")
                log_lines.append(
                    f"✅ **{path.name}**: {n_text} text block(s), {n_table} table(s), "
                    f"{n_ocr} OCR page(s) → {len(chunks)} chunks"
                )
            except Exception as e:
                log_lines.append(f"❌ **{path.name}**: failed — {e}")

        progress.progress(1.0, text="Building retriever...")
        st.session_state.chunks = all_chunks
        st.session_state.ingest_log = log_lines

        all_points = qdrant_client.scroll(
            collection_name=current_collection,
            limit=10_000,
            with_payload=True,
        )[0]
        from pipeline.chunker import Chunk as _Chunk
        full_corpus_chunks = [
            _Chunk(
                text=p.payload.get("page_content", "") or "",
                content_type=p.payload.get("metadata", {}).get("content_type", "text"),
                source=p.payload.get("metadata", {}).get("source", "unknown"),
                page=p.payload.get("metadata", {}).get("page"),
                table_id=p.payload.get("metadata", {}).get("table_id"),
            )
            for p in all_points
        ]

        if full_corpus_chunks:
            bm25 = build_bm25_retriever(full_corpus_chunks)
            hybrid = build_hybrid_retriever(vectorstore, bm25)
            reranker = build_reranker(top_n=5)
            st.session_state.hybrid_retriever = hybrid
            st.session_state.reranker = reranker
            st.session_state.agent = build_agent(hybrid, reranker=reranker, model_name=MODEL_NAME)
            progress.progress(1.0, text="Done.")
        else:
            st.warning("No chunks are indexed yet — check the ingestion log below.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if st.session_state.ingest_log:
    st.subheader("Ingestion log")
    for line in st.session_state.ingest_log:
        st.markdown(line)
    if st.session_state.chunks:
        n_table_chunks = sum(1 for c in st.session_state.chunks if c.content_type == "table")
        n_text_chunks = sum(1 for c in st.session_state.chunks if c.content_type == "text")
        st.info(f"Total: {len(st.session_state.chunks)} chunks "
                f"({n_text_chunks} text, {n_table_chunks} table)")
        with st.expander("Preview a few chunks"):
            for c in st.session_state.chunks[:5]:
                st.markdown(f"**[{c.content_type}]** `{c.source}` (page {c.page})")
                st.text(c.text[:500])
                st.divider()

# Step 2: ask questions
st.subheader("2. Ask a question")

if st.session_state.agent is None:
    st.caption("Run ingestion above first.")
else:
    with st.form(key="ask_question_form"):
        question = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        with st.spinner("Thinking..."):
            try:
                result = ask_with_trace(
                    st.session_state.agent, question,
                    st.session_state.hybrid_retriever,
                    reranker=st.session_state.reranker,
                )
                st.markdown("### Answer")
                st.markdown(result["answer"])

                if show_debug:
                    with st.expander(
                        f"🔍 Retrieval debug — {len(result['retrieved_chunks'])} chunk(s) "
                        f"retrieved for this question",
                        expanded=False,
                    ):
                        if not result["retrieved_chunks"]:
                            st.warning("No chunks were retrieved at all for this question — "
                                       "the answer above (if any) is NOT grounded in your "
                                       "documents.")
                        for i, c in enumerate(result["retrieved_chunks"], start=1):
                            st.markdown(f"**Chunk {i}** — `{c['header']}`")
                            st.text(c["text"][:800])
                            st.divider()
            except Exception as e:
                st.error(f"Agent call failed: {e}")