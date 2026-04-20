# Local Knowledge Search Engine — Design

**Date:** 2026-04-16  
**Status:** Approved

---

## Summary

A fully local, offline semantic search engine for Markdown, CSV, and plain text files.
No LLM, no cloud API. A single Python package exposes both a REST API and a CLI,
both sharing the same core library.

---

## Architecture

```
knowledge/
├── core/
│   ├── parser.py       # read .md, .csv, .txt → text chunks
│   ├── embedder.py     # load Qwen model, embed text
│   ├── index.py        # ChromaDB read/write (add, delete, query)
│   └── searcher.py     # embed query → vector search → ranked results
├── api/
│   └── server.py       # FastAPI: POST /index, POST /search, DELETE /document
└── cli/
    └── main.py         # `knowledge index <path>` / `knowledge search "<query>"`
```

**Indexing flow:**
```
files on disk → parser (chunk) → embedder (Qwen) → ChromaDB (persist)
```

**Search flow:**
```
query string → embedder (Qwen) → ChromaDB similarity search → ranked chunks + metadata
```

The CLI and API both call `core/` directly — no HTTP round-trip for local CLI use.

---

## Data Model

Each chunk stored in ChromaDB carries:

| Field | Description |
|---|---|
| `id` | `{file_path}:{chunk_index}` — unique, enables safe re-indexing |
| `text` | Raw chunk content |
| `file_path` | Absolute path to the source file |
| `file_type` | `md`, `csv`, or `txt` |
| `chunk_index` | Chunk position within the document |
| `indexed_at` | ISO 8601 timestamp |

---

## API Contract

```
POST /index
  body:     { "path": "/path/to/dir-or-file" }
  response: { "indexed": 42, "skipped": 3 }

POST /search
  body:     { "query": "...", "top_k": 10, "file_type": "md" }  # file_type optional
  response: [ { "text": "...", "file_path": "...", "score": 0.87, "chunk_index": 2 } ]

DELETE /document
  body:     { "file_path": "/path/to/file.md" }
  response: { "deleted": 5 }
```

---

## CLI Commands

```bash
knowledge index ./notes               # index a folder
knowledge index ./notes/file.md       # index a single file
knowledge search "how to deploy"      # top 5 results
knowledge search "query" --top-k 10   # more results
knowledge search "query" --type md    # filter by file type
knowledge delete ./notes/old.md       # remove from index
```

---

## Chunking Strategy

| File type | Strategy |
|---|---|
| `.md` | Split by heading (`#`, `##`), then by paragraph |
| `.txt` | Sliding window — 400 tokens, 50 token overlap |
| `.csv` | One chunk per row: `col1: val1, col2: val2, ...` |

---

## Tech Stack

| Concern | Library |
|---|---|
| Embeddings | `sentence-transformers` + `Qwen/Qwen3-Embedding` |
| Vector store | `chromadb` (embedded, file-based, no server needed) |
| API server | `fastapi` + `uvicorn` |
| CLI | `typer` |
| Markdown parsing | `markdown-it-py` |
| CSV / text parsing | Python stdlib |
| Config | `pydantic-settings` + `.env` |
| Tests | `pytest` |

**Embedding model notes:**
- `Qwen/Qwen3-Embedding` via `sentence-transformers`, 1024-dim vectors
- Downloaded once, cached at `~/.cache/huggingface/`
- GPU used automatically if available; falls back to CPU

---

## Configuration (`.env`)

```
CHROMA_DIR=~/.local/share/knowledge/chroma
EMBED_MODEL=Qwen/Qwen3-Embedding
CHUNK_SIZE=400
CHUNK_OVERLAP=50
API_PORT=8000
```

---

## Out of Scope

- LLM answer generation (search returns chunks, not synthesized answers)
- Web UI
- Authentication
- Auto-watching / live re-indexing on file changes
- Remote/cloud file sources
