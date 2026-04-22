# Local Knowledge Search Engine — Design

**Date:** 2026-04-16  
**Last updated:** 2026-04-22  
**Status:** Live

---

## Summary

A fully local, offline semantic search engine for Markdown, CSV, JSON, plain-text, and HTML files.
No cloud API. A single Python package exposes both a REST API and a CLI, both sharing the same `core/` library.

---

## Architecture

```
knowledge/
├── core/
│   ├── parser.py        # read .md, .csv, .txt, .json, .html → text chunks
│   ├── embedder.py      # load GGUF model via llama-cpp-python, embed text
│   ├── index.py         # ChromaDB read/write (add, delete, query, sample_chunks)
│   ├── searcher.py      # full search pipeline + intent gate + wiki management
│   ├── reranker.py      # optional LLM re-ranking via llama-cpp-python
│   └── domain_wiki.py   # DomainWiki: generate, embed, intent check, validate
├── api/
│   └── server.py        # FastAPI: POST /index, POST /search, DELETE /document
├── cli/
│   └── main.py          # `knowledge index/search/delete/wiki` commands
└── config.py            # pydantic-settings, all paths and thresholds
```

**Indexing flow:**
```
files on disk → parser (chunk) → embedder (GGUF/HF) → ChromaDB (persist)
```

**Search flow (four stages):**
```
query
  → [1] intent gate  (embedding vs domain wiki + keyword block list)
  → [2] vector retrieval  (ChromaDB cosine similarity, top-K*4 candidates)
  → [3] hybrid re-rank  (RRF: vector rank + BM25 keyword score)
  → [4] LLM re-rank  (optional, --rerank flag)
  → ranked results
```

Stage 1 returns `[]` immediately for out-of-scope queries, skipping stages 2–4.  
Stage 3 is on by default (`--no-hybrid` to skip).  
Stage 4 is opt-in (`--rerank`).

---

## Data Model

Each chunk stored in ChromaDB carries:

| Field | Description |
|---|---|
| `id` | `{file_path}:{chunk_index}` — unique, enables safe re-indexing |
| `text` | Raw chunk content |
| `file_path` | Absolute path to the source file |
| `file_type` | `md`, `csv`, `txt`, `json`, `html` |
| `chunk_index` | Chunk position within the document |
| `indexed_at` | ISO 8601 timestamp |

Domain wiki files live outside ChromaDB:

| File | Description |
|---|---|
| `domain-data/domain.md` | Human-readable markdown wiki (committed to git) |
| `domain-data/domain_emb.json` | Cached wiki embedding + keyword block list (git-ignored) |

---

## API Contract

```
POST /index
  body:     { "path": "/path/to/dir-or-file" }
  response: { "indexed": 42, "skipped": 3 }

POST /search
  body:     {
              "query": "...",
              "top_k": 5,
              "file_type": "md",        # optional filter
              "hybrid": true,           # default on
              "rerank": false,          # default off
              "min_score": 0.45,        # raw cosine threshold
              "intent_check": true      # default on
            }
  response: [ { "text": "...", "file_path": "...", "score": 0.87, "chunk_index": 2, ... } ]

DELETE /document
  body:     { "file_path": "/path/to/file.md" }
  response: { "deleted": 5 }
```

---

## CLI Commands

```bash
# Indexing
knowledge index ./notes                   # index a folder
knowledge index ./notes/file.md           # index a single file
knowledge index ./notes --update-wiki     # index + regenerate domain wiki

# Searching
knowledge search "how to deploy"          # top 5, hybrid on, intent on
knowledge search "query" --top-k 10       # more results
knowledge search "query" --type md        # filter by file type
knowledge search "query" --no-hybrid      # pure vector search
knowledge search "query" --rerank         # LLM re-rank (slower, best quality)
knowledge search "query" --no-intent      # bypass intent gate

# Index management
knowledge delete ./notes/old.md           # remove from index

# Domain wiki
knowledge wiki generate                   # LLM → domain-data/domain.md
knowledge wiki show                       # print current wiki
knowledge wiki path                       # print file path (shell scripting)
knowledge wiki check "query"              # dry-run intent gate, show score
knowledge wiki validate                   # audit block list, detect conflicts
```

---

## Chunking Strategy

| File type | Strategy |
|---|---|
| `.md` | Split by heading (`#`, `##`), then by paragraph |
| `.txt` | Sliding window — 400 tokens, 50 token overlap |
| `.csv` | One chunk per row: `col1: val1, col2: val2, ...` |
| `.json` | One chunk per array item or top-level object |
| `.html` / `.htm` | Tags stripped (BeautifulSoup), then sliding window |

---

## Tech Stack

| Concern | Library |
|---|---|
| Embeddings | `llama-cpp-python` + Qwen3-Embedding-0.6B GGUF (local) |
| Vector store | `chromadb` (embedded, file-based, no server needed) |
| Keyword search | `rank_bm25` + `jieba` (Chinese tokenisation) |
| LLM re-ranking / wiki generation | `llama-cpp-python` + Qwen2.5-Coder-1.5B GGUF |
| API server | `fastapi` + `uvicorn` |
| CLI | `typer` |
| HTML parsing | `beautifulsoup4` + `lxml` |
| Config | `pydantic-settings` + `.env` |
| Tests | `pytest` |

**Embedding model notes:**
- `Qwen3-Embedding-0.6B-Q8_0.gguf` via `llama-cpp-python`, 1024-dim vectors
- GGUF files placed in `models/` (git-ignored)
- GPU used automatically on Apple Silicon (Metal); falls back to CPU
- Alternatively, pass a HuggingFace repo id (e.g. `Qwen/Qwen3-Embedding-0.6B`) for auto-download

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `models/Qwen3-Embedding-0.6B-Q8_0.gguf` | Embedding model — GGUF path or HF repo id |
| `LLM_MODEL` | `models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` | LLM for wiki generation and re-ranking |
| `CHROMA_DIR` | `./chroma` | Vector DB storage path |
| `DOMAIN_WIKI_PATH` | `./domain-data/domain.md` | Domain wiki file |
| `DOMAIN_EMB_PATH` | `./domain-data/domain_emb.json` | Cached wiki embedding |
| `INTENT_THRESHOLD` | `0.25` | Minimum cosine similarity to pass the intent gate |
| `CHUNK_SIZE` | `400` | Words per chunk (txt, html) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `API_PORT` | `8000` | REST API listen port |

---

## Score Interpretation

| Mode | Score meaning |
|---|---|
| `--no-hybrid` | Raw cosine similarity (0–1); ~0.65 is a strong match |
| default (hybrid) | RRF-normalised (0–1); top result is always 1.0 |
| `--rerank` | LLM relevance score normalised to 0–1 |

`min_score` (default 0.45) filters individual results by raw cosine similarity before RRF normalisation, preventing weak candidates from being boosted by hybrid scoring.

---

## Out of Scope

- LLM answer generation (search returns chunks, not synthesised answers)
- Web UI
- Authentication
- Auto-watching / live re-indexing on file changes
- Remote / cloud file sources
- PDF, Word, or Office document support
