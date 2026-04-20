# Local Knowledge Search

Fully local semantic search over Markdown, CSV, JSON, plain-text, and HTML files.
No cloud API required — embeddings and re-ranking run entirely on-device via GGUF models.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env        # review defaults, adjust paths if needed

knowledge index ./my-notes  # index a folder
knowledge search "query"    # search
```

## Models

Place GGUF files in the `models/` folder (already in `.gitignore`):

| File | Purpose |
|------|---------|
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | Embeddings (required) |
| `Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` | Re-ranking LLM (optional, used with `--rerank`) |

Download from HuggingFace:
- [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF)

**Apple Silicon:** Install llama-cpp-python with Metal acceleration:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Alternatively**, use a HuggingFace SentenceTransformer by setting `EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B` in `.env` (requires internet for first download).

## CLI Usage

```bash
knowledge index ./my-notes              # index a folder (md, txt, csv, json, html)
knowledge index ./notes/article.html    # index a single file

knowledge search "query here"           # semantic search (default top 5)
knowledge search "query" --top-k 10     # return more results
knowledge search "query" --type md      # filter by file type
knowledge search "query" --rerank       # re-rank with local LLM (slower, more accurate)

knowledge delete ./my-notes/old.md      # remove a file from the index
```

## API Server

```bash
python -m knowledge.api
# Runs at http://localhost:8000
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/index` | Index a file or directory |
| `POST` | `/search` | Semantic search |
| `DELETE` | `/document` | Remove a file from the index |

`POST /search` body:
```json
{ "query": "rainy day hiking", "top_k": 5, "file_type": null, "rerank": false }
```

## Configuration

Copy `.env.example` to `.env` and edit as needed. All values can also be set as environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `models/Qwen3-Embedding-0.6B-Q8_0.gguf` | Embedding model — GGUF path or HF repo id |
| `LLM_MODEL` | `models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` | Re-ranking LLM — GGUF path |
| `CHROMA_DIR` | `./chroma` | Vector DB storage — project-local by default |
| `CHUNK_SIZE` | `400` | Words per chunk (txt, html) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `API_PORT` | `8000` | REST API port |

To share a single index across projects, point `CHROMA_DIR` at a fixed location:
```
CHROMA_DIR=~/.local/share/knowledge/chroma
```

> **Note:** If you switch embedding models, delete or relocate `CHROMA_DIR` and re-index — vector dimensions must match.

## Supported File Types

| Extension | Chunking strategy |
|-----------|------------------|
| `.txt` | Fixed-size word windows with overlap |
| `.md` | Split on `#` headings |
| `.csv` | One chunk per row |
| `.json` | One chunk per array item or object |
| `.html` / `.htm` | Tags stripped, then fixed-size word windows |

## Understanding Search Scores

Scores are cosine similarity values (0–1). Raw embedding similarity is naturally modest:

| Score | Meaning |
|-------|---------|
| 0.7+ | Near-identical or paraphrased content |
| 0.4–0.7 | Clearly related topic |
| 0.2–0.4 | Loosely related |
| < 0.2 | Likely unrelated |

ChromaDB always returns the requested `top_k` results regardless of score. Use `--rerank` to improve ordering when results feel mixed.

## Tests

```bash
pytest -v
# Parser/reranker tests run immediately (no model needed).
# Embedding/searcher tests are skipped if the GGUF model is not in models/.
```

## Docs

- [`docs/designs/`](docs/designs/) — architecture decisions
- [`docs/plans/`](docs/plans/) — implementation plans
- [`docs/archive/`](docs/archive/) — superseded earlier designs
