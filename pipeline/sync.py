"""
Content-hash-based sync: avoids re-embedding unchanged files and
avoids leaving stale chunks behind when a file's content changes.

The approach, at file level (not per-chunk):

  1. Hash the file's raw bytes (sha256).
  2. Look up whether a chunk from this source is already stored in
     Qdrant, and if so, what hash it was stored with (every chunk is
     tagged with `file_hash` metadata at insert time).
  3. Compare:
     - No existing chunks for this source -> new file, ingest normally.
     - Existing hash == new hash -> unchanged, SKIP re-embedding entirely.
     - Existing hash != new hash -> file was edited since last run;
       delete all of its old chunks first, then re-ingest fresh ones.
       (Delete-then-reinsert, not update-in-place, because chunk
       boundaries can shift when content changes -- there's no stable
       mapping from "old chunk 3" to "new chunk 3".)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from qdrant_client.models import Filter, FieldCondition, MatchValue


def compute_file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_stored_hash_for_source(client, collection_name: str, source: str) -> str | None:
    """Returns the file_hash stored on an existing chunk for this
    source, or None if no chunks from this source are indexed yet.
  """
    if not client.collection_exists(collection_name):
        return None
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
        limit=1,
        with_payload=True,
    )
    if not results:
        return None
    return results[0].payload.get("metadata", {}).get("file_hash")


def delete_source_chunks(client, collection_name: str, source: str) -> int:
    """Deletes all indexed chunks for a given source file. Returns the
    count deleted (best-effort -- Qdrant's delete-by-filter doesn't
    always report an exact count, so this is approximate)."""
    if not client.collection_exists(collection_name):
        return 0
    existing, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
        limit=10_000,
        with_payload=False,
    )
    if not existing:
        return 0
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
    )
    return len(existing)


class SyncDecision:
    SKIP_UNCHANGED = "skip_unchanged"
    NEW_FILE = "new_file"
    UPDATED_FILE = "updated_file"


def decide_sync_action(client, collection_name: str, path: str | Path) -> tuple[str, str]:
    """Returns (decision, file_hash). Call this BEFORE loading/chunking
    a file so you can skip the expensive parse+embed work entirely on
    the unchanged-file fast path."""
    file_hash = compute_file_hash(path)
    source = str(path)
    stored_hash = get_stored_hash_for_source(client, collection_name, source)

    if stored_hash is None:
        return SyncDecision.NEW_FILE, file_hash
    if stored_hash == file_hash:
        return SyncDecision.SKIP_UNCHANGED, file_hash
    return SyncDecision.UPDATED_FILE, file_hash