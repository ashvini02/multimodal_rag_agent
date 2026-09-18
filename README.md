# Document Processing Pipeline for Multi-Format Documents

## Objective
This project is designed to process and query a mixed corpus of documents with a strong focus on the actual failure modes that appear in real-world document pipelines:

- PDF files with selectable text
- scanned/image-based PDFs requiring OCR and vision fallback
- PPT files with structured elements and image-heavy slides
- Excel/CSV files with tables and sheet-based data

---

## Executive summary
The core idea is simple but important:

1. Ingest files by format, not by one generic parser.
2. Normalize everything into a shared intermediate structure so downstream logic is consistent.
3. Use OCR/vision only when a document is truly image-heavy or text is not extractable.
4. Chunk by content type so tables and prose(text) are treated differently.
5. Retrieve using hybrid search (BM25 + dense vector search) and re-rank before the LLM sees the context.
6. Force the final answer to be grounded in retrieved chunks only, so the model does not answer from its own background knowledge.
7. Keep the active corpus scoped to the current session, while still avoiding re-embedding the same file across sessions when the content hash matches.

---

## Architecture overview

```text
                         ┌─────────────────────┐
                         │   User upload UI    │
                         │ streamlit_app.py    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Session-scoped upload flow      │
                    │  one collection per session      │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────────┐
                     │      Multi-format ingestion      │
                     │  ingestion/loaders.py           │
                     │  ingestion/ocr.py               │
                     └──────────────┬─────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────┐
                     │   Normalized blocks            │
                     │  text | table | image_ocr      │
                     └──────────────┬─────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────┐
                     │   Chunking layer               │
                     │   pipeline/chunker.py          │
                     │   char-based + table-aware     │
                     └──────────────┬─────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────┐
                     │   Embedding + Qdrant index       │
                     │   pipeline/index.py              │
                     │   + hash-based check-up/reuse    │
                     └──────────────┬───────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────┐
                     │    Hybrid retrieval            │
                     │ BM25 + dense + reranker        │
                     │ pipeline/reranker.py           │
                     └──────────────┬─────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────┐
                     │   Grounded Q&A agent           │
                     │   agent/rag_agent.py           │
                     │   deepagents orchestration     │
                     └────────────────────────────────┘
```

---

## Project structure

```text
.
├── agent/
│   └── rag_agent.py              # orchestrator + retrieval-grounding logic
├── ingestion/
│   ├── loaders.py                # PDF / PPT / Excel loaders
│   ├── ocr.py                   # OCR + vision fallback
│   └── __init__.py
├── pipeline/
│   ├── chunker.py               # content-aware chunking and table handling
│   ├── index.py                 # embeddings, Qdrant, BM25 + hybrid retriever
│   ├── ingest.py                # directory ingestion orchestration
│   ├── reranker.py              # rerank step
│   ├── sync.py                  # file-hash sync logic
│   └── __init__.py
├── main.py                      # CLI run path
├── streamlit_app.py             # Streamlit upload + question UI
├── requirements.txt             # dependencies
├── README.md                    # project documentation
├── .env.example                 # environment configuration (if present)
```

---

## Design strategy

### 1. Format-specific extraction instead of one generic parser
A single generic parser is tempting, but it introduces a common failure mode: it may treat text-heavy documents as images or degrade table extraction quality. This project intentionally uses dedicated loaders for each file type.

- PDF: `pdfplumber` for text and table extraction
- scanned PDF: page-level OCR or vision fallback if extractable text is missing
- PPT: `python-pptx` for slide text and tables
- Excel/CSV: `pandas` and `openpyxl` for structured sheet extraction

This is a strong decision because it makes each file type’s extraction strategy explicit, testable, and debuggable.

### 2. Normal PDFs vs scanned PDFs are handled differently
A major part of the assignment is the scanned PDF requirement. The implementation explicitly checks whether a PDF page has a usable text layer.

- If normal text is available, the page is treated as a standard PDF text block.
- If the text layer is empty or poor, the page is treated as a scanned document and sent through OCR or visual extraction.

This is the right architecture because scanned PDFs are not just a parser issue; they are a different information source and must be processed differently.

### 3. Table-aware chunking is necessary
Tables are not chunks of prose. A table row is a different semantic unit than a paragraph. This project treats tables as first-class content and preserves their structure.

- table rows are kept as table blocks
- headers are repeated in fragments
- table chunks are not mixed with surrounding narrative text
- content-type metadata is retained for downstream decisions

This is a better design than blindly chunking all content by character count, because it prevents a table fragment from being compressed into surrounding prose and losing meaning.

### 4. Retrieval is hybrid, not vector-only
The retrieval layer uses both sparse and dense access patterns:

- BM25 catches exact keyword matches and rare tokens
- dense vector search catches semantic relationships
- a reranker then scores the combined result set down to the highest-value chunks

This matters because a purely dense retriever often misses exact names, IDs, table values, and code-like labels that are critical in business documents.

### 5. The final answer is grounded, not guessed
A major design requirement for this assignment is that the system should not hallucinate or answer from general knowledge when the uploaded content does not cover a query.

The agent logic is designed to:

- retrieve context first
- filter out irrelevant chunks
- reject weak/no-overlap answers
- explicitly return a “not in the uploaded content” answer when the query is unsupported

This is a strong strategy for showing correctness and control, especially in an evaluation environment where they will check whether the answer is grounded and not just plausible.

### 6. Session-scoped document corpus is a correctness feature
The project intentionally keeps the active corpus tied to the current session collection instead of letting stale files from earlier uploads remain in the answer space. This is critical because a user should not be asked a question about a document that was uploaded in a different session or an older batch unless that file is deliberately reused by hash.

This is a real product-quality constraint and shows thoughtful system design rather than a simplistic “global collection forever” approach.

### 7. Hash-based re-use avoids wasted work without causing leakage
The project uses SHA-256 checks to avoid re-embedding unchanged files and to reuse previously indexed file content when the hash matches.

This is useful because it reduces repeat work and keeps the system efficient without breaking the session isolation rule. The key implementation question was always: where to store the hash relation, and how to keep the scope correct. This project uses a session-aware collection pattern plus hash metadata checks to meet both goals.

---

## Critical technical details and trade-offs

### Why not a single generic parser?
The system intentionally avoids a single “one-size-fits-all” parser because real-world document corpora are messy. A generic parser may do well on standard PDFs but often falls apart on scanned pages, table-heavy sheets, and presentation files with mixed visual layout.

### Why local embeddings and local reranking?
This keeps the system self-contained and reduces dependency risk. It also makes the pipeline more reproducible and more suitable for demo/evaluation in a constrained environment.

### Why hybrid retrieval instead of pure semantic search?
Because exact values, codes, labels, and business terms are often not captured well by embeddings alone. BM25 acts as the exact-match safety net.

### Why a deep-agent orchestration layer?
Because the task is not just retrieval; it is grounded answer synthesis. The agent adds a reasoning layer that can: retrieve, summarize, inspect chunks, and answer with source-aware reasoning instead of bluntly returning raw text.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
# venv\Scripts\Activate.ps1

# Windows (cmd)
# venv\Scripts\activate.bat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (either export them, or put them in a .env file --
#    both main.py and streamlit_app.py call load_dotenv() on startup)
export OPENAI_API_KEY=...       # generation (gpt-4o-mini) + OCR vision fallback -- required

# 4. Start Qdrant (vector store) locally via Docker
docker run -p 6333:6333 qdrant/qdrant
```

## Run — CLI

```bash
python main.py ./sample_docs "What was the Q3 revenue by region?"
```

Note: the CLI uses a single shared "docs" collection across runs, not session-scoped isolation.

## Run — Streamlit UI

```bash
streamlit run streamlit_app.py
```

Upload files, click "Run ingestion", watch the per-file log (block/table/OCR counts, including when a file is skipped as unchanged), then ask questions in the box below. Answers are scoped only to files uploaded in the current session — click "Start fresh session" to explicitly clear isolation and start over. Enable "Show retrieval debug panel" in the sidebar to see exactly which chunks were retrieved for each question, including an explicit warning when zero chunks were retrieved at all.

---