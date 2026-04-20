# LangChain Loaders + LLM Re-ranker — Design

**Date:** 2026-04-20
**Status:** Approved

## Goal

Replace the custom file parser with LangChain document loaders for generic multi-format support, add an LLM-powered re-ranking layer using the local Qwen2.5-Coder-1.5B GGUF model, and delete all hardcoded domain logic (`catalog_filters.py`).

## Architecture

```
Files on disk
     ↓
[LangChain Loaders]         knowledge/core/loader.py   (replaces parser.py)
     ↓ LangChain Documents → Chunks
[ChromaDB + GGUF Embedder]  knowledge/core/index.py    (unchanged)
     ↓ top-K chunks
[LangChain LLMChain]        knowledge/core/reranker.py (new)
     ↓ re-ranked results + optional summary
[API / CLI]                                             (--rerank flag added)
```

## Components

### `knowledge/core/loader.py` — replaces `parser.py`

LangChain loader selected by file extension:

| Extension | Loader |
|-----------|--------|
| `.md`, `.txt` | `TextLoader` |
| `.csv` | `CSVLoader` |
| `.json` | `JSONLoader` (jq_schema=".") |
| `.html` | `BSHTMLLoader` |

Each `Document.page_content` is converted to the existing `Chunk` dataclass so downstream code (`index.py`, `searcher.py`) stays unchanged.

### `knowledge/core/reranker.py` — new

`Reranker` class:
1. Accepts `(query: str, results: list[SearchResult])`
2. Builds a structured prompt listing each chunk with its index
3. Calls `Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` via `llama-cpp-python`
4. Parses LLM scores (0–10 per chunk) and re-sorts results
5. Returns re-sorted `list[SearchResult]` with optional one-sentence summary

Fallback: if model file is missing or LLM call fails, returns results in original vector-score order with a logged warning.

### Deleted

- `knowledge/core/catalog_filters.py`
- `tests/core/test_catalog_filters.py`

### API & CLI changes

Both gain `--rerank / --no-rerank` (default: `--no-rerank`).

When `--rerank` is enabled:
- Search fetches `top_k * 2` candidates from ChromaDB
- Passes them through `Reranker`
- Returns top `top_k` by LLM score

## Data Flow

### Indexing

```
index_path(path)
  → loader selected by extension
  → Document.page_content → Chunk list
  → embed + upsert into ChromaDB
```

### Search with re-ranking

```
search(query, top_k=5, rerank=True)
  → ChromaDB query with n_results = top_k * 2
  → Reranker.rerank(query, candidates)
  → return top_k sorted by LLM score
```

## Error Handling

- **Unsupported extension in directory scan** — skip silently, log at DEBUG
- **Unsupported extension explicit call** — raise `ValueError`
- **LLM model file missing** — fall back to vector order, log WARNING
- **LLM parse failure** — fall back to vector order, log WARNING
- **Malformed HTML/JSON** — loader skips file, logs WARNING

## Testing

| Test file | What it covers |
|-----------|---------------|
| `tests/core/test_loader.py` | txt, md, csv, json, html loading; unsupported raises |
| `tests/core/test_reranker.py` | re-ordering via mocked LLM; graceful fallback |
| `tests/core/test_catalog_filters.py` | **deleted** |
| All existing API + CLI tests | pass unchanged (stable interface) |

## Dependencies Added

- `langchain-community` — document loaders
- `langchain` — LLMChain, PromptTemplate
- `langchain-core` — base interfaces
- `beautifulsoup4` — for BSHTMLLoader
- `lxml` — HTML parser backend

## Not in scope

- Multi-turn conversation / memory
- Domain-specific query expansion
- PDF, Word, or Office document support
