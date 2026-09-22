"""
Chunking, with content-type-aware (structure-aware) strategy.

Core design decision: tables are NEVER chunked by naive character
count either. They are chunked by ROW, with the header row repeated
in every resulting chunk, so that:
  (a) a table chunk is self-contained and interpretable on its own
      even if retrieved without its neighbors, and
  (b) a table too large for one chunk still produces N clean,
      independently-meaningful chunks instead of being cut at an
      arbitrary character offset that might land mid-row.

Plain text blocks use recursive character splitting (paragraph -> sentence -> word fallback), 
a separate code path -- text and tables are never mixed in the same chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.loaders import Block

# Defaults sized for character counts. ~2000 chars =~ 500 tokens,
# matching the token-based defaults used earlier in this project,
# so retrieval behavior stays comparable after the switch.
DEFAULT_MAX_CHARS = 500
DEFAULT_OVERLAP_CHARS = 100


@dataclass
class Chunk:
    text: str
    content_type: str
    source: str
    page: int | None = None
    table_id: str | None = None
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def char_len(text: str) -> int:
    return len(text)


# ----- Text chunking (recursive, structure-aware) -----
def chunk_text_block(block: Block, max_chars: int = DEFAULT_MAX_CHARS,
                      overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[Chunk]:
    separators = ["\n\n", "\n", ". ", " "]
    pieces = recursive_split(block.text, separators, max_chars)
    pieces = apply_overlap(pieces, overlap_chars)

    return [
        Chunk(
            text=piece,
            content_type="text",
            source=block.source,
            page=block.page,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]


def recursive_split(text: str, separators: list[str], max_chars: int) -> list[str]:
    if char_len(text) <= max_chars:
        return [text] if text.strip() else []

    if not separators:
        # hard split by character count.
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    sep, *rest = separators
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if char_len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # this single part might itself be too big -- recurse with next separator
            if char_len(part) > max_chars:
                chunks.extend(recursive_split(part, rest, max_chars))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def apply_overlap(pieces: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(pieces) <= 1:
        return pieces
    result = [pieces[0]]
    for i in range(1, len(pieces)):
        prev = pieces[i - 1]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        result.append(tail + "\n" + pieces[i])
    return result


# ----- Table chunking (row-based, header-repeated) -----
def chunk_table_block(block: Block, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    rows = block.extra.get("raw_rows")
    if not rows or len(rows) < 2:
        # No structured rows available (shouldn't normally happen) --
        # fall back to treating it as a single opaque chunk.
        return [Chunk(text=block.text, content_type="table", source=block.source,
                       page=block.page, table_id=block.table_id, chunk_index=0)]

    header, *body_rows = rows
    header_md = rows_to_markdown_pair(header)
    header_chars = char_len(header_md)

    chunks: list[Chunk] = []
    current_rows: list[list[str]] = []
    current_chars = header_chars

    def flush(idx: int):
        if not current_rows:
            return
        md = rows_to_markdown_table(header, current_rows)
        chunks.append(Chunk(
            text=md,
            content_type="table",
            source=block.source,
            page=block.page,
            table_id=block.table_id,
            chunk_index=idx,
            metadata={
                "row_start": idx,
                "is_table_fragment": len(body_rows) > 0,
            },
        ))

    for row in body_rows:
        clean_row = ["" if cell is None else str(cell) for cell in row]
        row_md_len = char_len(" | ".join(clean_row))
        if current_rows and (current_chars + row_md_len > max_chars):
            flush(len(chunks))
            current_rows = []
            current_chars = header_chars  # every new chunk restarts with header chunks
        current_rows.append(clean_row)
        current_chars += row_md_len

    flush(len(chunks))

    # Tag every chunk with total fragment count so a retriever/the agent downstream can know 
    # "this is part 2 of 5 of table X" and request neighboring fragments if the answer needs the full table.
    total = len(chunks)
    for c in chunks:
        c.metadata["fragment_of_total"] = total

    return chunks


def rows_to_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    clean_header = ["" if cell is None else str(cell) for cell in header]
    out = ["| " + " | ".join(clean_header) + " |",
           "| " + " | ".join(["---"] * len(clean_header)) + " |"]
    for row in rows:
        clean_row = ["" if cell is None else str(cell) for cell in row]
        clean_row = clean_row + [""] * (len(clean_header) - len(clean_row))
        out.append("| " + " | ".join(clean_row[:len(clean_header)]) + " |")
    return "\n".join(out)


def rows_to_markdown_pair(header: list[str]) -> str:
    clean_header = ["" if cell is None else str(cell) for cell in header]
    return ("| " + " | ".join(clean_header) + " |\n"
            "| " + " | ".join(["---"] * len(clean_header)) + " |")



def chunk_block(block: Block, max_chars: int = DEFAULT_MAX_CHARS,
                 overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[Chunk]:
    if block.content_type == "table":
        return chunk_table_block(block, max_chars=max_chars)
    # "text" and "image_ocr" blocks both go through the same text chunker --
    # OCR output is just text once extracted.
    return chunk_text_block(block, max_chars=max_chars, overlap_chars=overlap_chars)


def chunk_all(blocks: list[Block], max_chars: int = DEFAULT_MAX_CHARS,
              overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for block in blocks:
        all_chunks.extend(chunk_block(block, max_chars=max_chars, overlap_chars=overlap_chars))
    return all_chunks