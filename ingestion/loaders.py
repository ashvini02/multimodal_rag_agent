"""
Format-specific document loaders.

Design decision: instead of a single generic parser (e.g. `unstructured`) for every format, 
each file type gets its own loader. This avoids the common failure mode where a 
generic parser treats a text-heavy PDF page as an image.

Every loader returns a list of `Block` objects -- a normalized intermediate representation 
with an explicit `content_type`("text" | "table" | "image_ocr"). 
Downstream chunking treats these types differently, which is the key to handle tables and scanned pages correctly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ContentType = Literal["text", "table", "image_ocr"]


@dataclass
class Block:
    """A single structural unit extracted from a document."""
    content_type: ContentType
    text: str                       # plain text OR markdown-rendered table
    source: str                     # file name
    page: int | None = None         # page/slide/sheet number, 1-indexed
    table_id: str | None = None     # set for content_type == "table"
    extra: dict = field(default_factory=dict)


# ----- PDF -----
def load_pdf(path: str | Path) -> list[Block]:
    """
    Loads a PDF using pdfplumber, which gives both TEXT and TABLE
    extraction in one pass (this is why it's preferred here over
    PyMuPDF alone -- PyMuPDF is faster for pure text but has weaker
    table structure detection).

    Pages with no extractable text (scanned pages) are flagged and
    handed off to the OCR/vision fallback in ingestion/ocr.py.
    """
    import pdfplumber
    from ingestion.ocr import ocr_page_image

    blocks: list[Block] = []
    path = str(path)

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            tables = page.extract_tables() or []

            if should_trigger_ocr(page, text, tables):
                ocr_text = ocr_page_image(page, source=path, page_num=page_num)
                if ocr_text:
                    blocks.append(Block(
                        content_type="image_ocr",
                        text=ocr_text,
                        source=path,
                        page=page_num,
                    ))
                elif text:
                    blocks.append(Block(
                        content_type="text",
                        text=text,
                        source=path,
                        page=page_num,
                    ))
                continue

            if text:
                blocks.append(Block(
                    content_type="text",
                    text=text,
                    source=path,
                    page=page_num,
                ))

            for t_idx, raw_table in enumerate(tables):
                md_table = rows_to_markdown(raw_table)
                blocks.append(Block(
                    content_type="table",
                    text=md_table,
                    source=path,
                    page=page_num,
                    table_id=f"{Path(path).stem}_p{page_num}_t{t_idx}",
                    extra={"raw_rows": raw_table},
                ))

    return merge_cross_page_tables(blocks)


# ----- PPTX -----
def load_pptx(path: str | Path) -> list[Block]:
    """
    python-pptx: extract text frames, tables, and speaker notes
    (notes are often skipped by generic parsers but can carry
    real content). Slide images without text placeholders are
    sent to OCR if they contain a picture shape with no
    accompanying text -- a common case for slides that are
    entirely a screenshot/diagram.
    """
    from pptx import Presentation
    from ingestion.ocr import ocr_image_bytes

    prs = Presentation(str(path))
    blocks: list[Block] = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_text: list[str] = []
        has_text_content = False

        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                slide_text.append(shape.text_frame.text.strip())
                has_text_content = True

            if shape.has_table:
                rows = [[cell.text.strip() for cell in row.cells]
                        for row in shape.table.rows]
                blocks.append(Block(
                    content_type="table",
                    text=rows_to_markdown(rows),
                    source=str(path),
                    page=slide_num,
                    table_id=f"{Path(path).stem}_s{slide_num}",
                    extra={"raw_rows": rows},
                ))
                has_text_content = True

            # Picture shape with no other text on the slide --> likely a screenshot/diagram-only slide. OCR it.
            if shape.shape_type == 13 and not has_text_content:  # 13 == PICTURE
                try:
                    image_bytes = shape.image.blob
                    ocr_text = ocr_image_bytes(image_bytes, source=str(path), page_num=slide_num)
                    if ocr_text:
                        blocks.append(Block(
                            content_type="image_ocr",
                            text=ocr_text,
                            source=str(path),
                            page=slide_num,
                        ))
                except Exception:
                    pass  # image extraction can fail on unusual embeds; don't crash ingestion

        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_text.append(f"[Speaker notes]: {notes}")

        if slide_text:
            blocks.append(Block(
                content_type="text",
                text="\n".join(slide_text),
                source=str(path),
                page=slide_num,
            ))

    return blocks


# ----- XLSX / CSV -----
def load_spreadsheet(path: str | Path) -> list[Block]:
    """
    Every sheet is treated as one (or more, if very wide/long) table
    block. Unlike PDF/PPTX tables, spreadsheet tables are the ENTIRE
    content -- so row-based chunking (table_chunking.py) is the
    primary chunking strategy for this file type.
    """
    import pandas as pd

    path = str(path)
    blocks: list[Block] = []

    if path.lower().endswith(".csv"):
        sheets = {"Sheet1": pd.read_csv(path, dtype=str, keep_default_na=False)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
        for name in sheets:
            sheets[name] = sheets[name].fillna("")

    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        rows = [list(df.columns)] + df.astype(str).values.tolist()
        blocks.append(Block(
            content_type="table",
            text=rows_to_markdown(rows),
            source=path,
            table_id=f"{Path(path).stem}_{sheet_name}",
            extra={"raw_rows": rows, "sheet_name": sheet_name},
        ))

    return blocks


# helper functions
def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render a list-of-lists table as a markdown table string.
    Markdown is used (rather than raw CSV-ish text) because LLMs
    parse markdown tables reliably and it's what you'll want to
    reproduce in the final answer anyway."""
    if not rows:
        return ""
    rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in row]
            for row in rows]
    header, *body = rows
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        row = row + [""] * (len(header) - len(row))  # pad ragged rows
        out.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(out)


def should_trigger_ocr(page, text: str, tables: list[list]) -> bool:
    """Treat image-heavy or badly garbled text PDFs as scanned pages,
    even when a low-quality text layer already exists."""
    if tables:
        return False

    images = getattr(page, "images", None) or []
    if images:
        page_area = max(float(getattr(page, "width", 0) or 0) * float(getattr(page, "height", 0) or 0), 1.0)
        image_area = 0.0
        for img in images:
            try:
                x0 = float(img.get("x0", 0))
                x1 = float(img.get("x1", 0))
                y0 = float(img.get("y0", 0))
                y1 = float(img.get("y1", 0))
                image_area += max(0.0, (x1 - x0) * (y1 - y0))
            except Exception:
                continue
        if image_area >= page_area * 0.6:
            return True

    if not text:
        return True

    # OCR text layers on scans are often broken, fragmented, and full of
    # obvious substitution errors like repeated letters and malformed words.
    words = [w for w in text.replace("\n", " ").split() if w]
    if len(words) < 20:
        return True

    weird_word_count = 0
    for word in words:
        letters = "".join(ch for ch in word if ch.isalpha())
        if len(letters) <= 2:
            weird_word_count += 1
            continue
        if any(ch.isdigit() for ch in word):
            weird_word_count += 1
            continue
        if sum(1 for ch in letters if ch in "aeiou") < 1 and len(letters) > 5:
            weird_word_count += 1
    return weird_word_count / max(len(words), 1) > 0.25


def merge_cross_page_tables(blocks: list[Block]) -> list[Block]:
    """
    Heuristic merge for tables that continue across a PDF page break:
    if a table block is immediately followed (in block order) by
    another table block on the very next page, with the SAME number
    of columns and no text block in between on the new page before
    the table, treat them as one logical table.

    This is a heuristic, not a guarantee -- flag it in your writeup
    as a known limitation and something you'd validate against a
    real corpus of multi-page tables before trusting it blindly.
    """
    merged: list[Block] = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        if current.content_type != "table":
            merged.append(current)
            i += 1
            continue

        # look ahead for a continuation table on the next page
        j = i + 1
        combined_rows = list(current.extra.get("raw_rows", []))
        while j < len(blocks):
            nxt = blocks[j]
            same_next_page = (current.page is not None and nxt.page == current.page + 1)
            is_table = nxt.content_type == "table"
            same_width = (is_table and nxt.extra.get("raw_rows") and
                          len(nxt.extra["raw_rows"][0]) == len(combined_rows[0]))
            if is_table and same_next_page and same_width:
                # drop the continuation's header row (assume row 0 repeats the header)
                combined_rows.extend(nxt.extra["raw_rows"][1:])
                current = current  # keep growing `current`
                j += 1
            else:
                break

        if j > i + 1:
            merged_block = Block(
                content_type="table",
                text=rows_to_markdown(combined_rows),
                source=current.source,
                page=current.page,
                table_id=current.table_id,
                extra={"raw_rows": combined_rows, "merged_from_pages": j - i},
            )
            merged.append(merged_block)
            i = j
        else:
            merged.append(current)
            i += 1

    return merged


LOADERS = {
    ".pdf": load_pdf,
    ".pptx": load_pptx,
    ".xlsx": load_spreadsheet,
    ".xls": load_spreadsheet,
    ".csv": load_spreadsheet,
}


def load_file(path: str | Path) -> list[Block]:
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"No loader registered for extension '{ext}'")
    return loader(path)