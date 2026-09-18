"""
Ingestion pipeline entrypoint: walks a directory of mixed-format
files and syncs each one into the vector store, using content-hash
checks (pipeline/sync.py) to skip unchanged files and clean up stale
chunks for edited ones -- see sync.py's docstring for the full logic.
"""
from __future__ import annotations

from pathlib import Path

from ingestion.loaders import load_file, LOADERS
from pipeline.chunker import chunk_all, Chunk
from pipeline.index import get_vectorstore, build_vectorstore
from pipeline.sync import decide_sync_action, delete_source_chunks, SyncDecision


def ingest_directory(directory: str | Path, max_chars: int = 500,
                      overlap_chars: int = 100,
                      collection_name: str = "docs",
                      qdrant_url: str = "http://localhost:6333") -> list[Chunk]:
    """
    Sync-aware ingestion: unchanged files are skipped without being
    re-parsed or re-embedded; edited files have their old chunks
    deleted before new ones are inserted; new files are ingested
    normally. Returns the chunks that were newly added or updated in
    this run.
    """
    directory = Path(directory)
    vectorstore, client = get_vectorstore(collection_name, qdrant_url)

    files = [p for p in directory.rglob("*") if p.suffix.lower() in LOADERS]
    print(f"Found {len(files)} supported files in {directory}")

    new_or_updated_chunks: list[Chunk] = []

    for path in files:
        decision, file_hash = decide_sync_action(client, collection_name, path)

        if decision == SyncDecision.SKIP_UNCHANGED:
            print(f"  [SKIP] {path.name}: unchanged since last ingestion")
            continue

        if decision == SyncDecision.UPDATED_FILE:
            deleted = delete_source_chunks(client, collection_name, str(path))
            print(f"  [UPDATE] {path.name}: content changed, removed {deleted} stale chunk(s)")

        try:
            blocks = load_file(path)
        except Exception as e:
            print(f"  [FAILED] {path.name}: {e}")
            continue

        chunks = chunk_all(blocks, max_chars=max_chars, overlap_chars=overlap_chars)
        build_vectorstore(chunks, collection_name=collection_name,
                                   qdrant_url=qdrant_url, file_hash=file_hash)
        new_or_updated_chunks.extend(chunks)

        n_text = sum(1 for b in blocks if b.content_type == "text")
        n_table = sum(1 for b in blocks if b.content_type == "table")
        n_ocr = sum(1 for b in blocks if b.content_type == "image_ocr")
        tag = "NEW" if decision == SyncDecision.NEW_FILE else "UPDATED"
        print(f"  [{tag}] {path.name}: {n_text} text block(s), {n_table} table(s), "
              f"{n_ocr} OCR page(s) -> {len(chunks)} chunks indexed")
        

    print(f"Newly indexed/updated chunks are: {len(new_or_updated_chunks)}")
    return new_or_updated_chunks