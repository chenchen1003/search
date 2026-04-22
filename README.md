# Local Knowledge Search

Fully local semantic search over Markdown, CSV, JSON, plain-text, and HTML files.
No cloud API required — embeddings and re-ranking run entirely on-device via GGUF models.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env          # review defaults, adjust paths if needed

knowledge index ./my-notes    # index a folder
knowledge wiki generate       # build domain wiki (improves relevance, optional)
knowledge search "query"      # search
```

### Ubuntu, Debian, and WSL

On these systems the distro Python is **externally managed** ([PEP 668](https://peps.python.org/pep-0668/)), so running `pip install` without a virtual environment fails with `externally-managed-environment`. Use a venv in the project directory (or any path you prefer):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Keep the venv activated when you run `knowledge` or `python -m knowledge.api`.

If `python3 -m venv` is missing:

```bash
sudo apt install python3-venv python3-full
```

## Models

Place GGUF files in the `models/` folder (already in `.gitignore`):

| File | Purpose |
|------|---------|
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | Embeddings — **required** |
| `Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` | LLM for wiki generation and re-ranking — **optional** |

Download from HuggingFace:
- [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF)

**Apple Silicon:** Install llama-cpp-python with Metal acceleration:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Alternatively**, use a HuggingFace SentenceTransformer by setting `EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B` in `.env` (requires internet for first download).

## CLI Usage

### Indexing

```bash
knowledge index ./my-notes              # index a folder (md, txt, csv, json, html)
knowledge index ./notes/article.html    # index a single file
knowledge index ./my-notes --update-wiki  # index + regenerate domain wiki in one step
knowledge delete ./my-notes/old.md      # remove a file from the index
```

After indexing, if a domain wiki already exists the command prints a reminder to refresh it.

### Searching

```bash
knowledge search "query here"           # semantic search, default top 5
knowledge search "query" --top-k 10     # return more results
knowledge search "query" --type md      # filter by file type
knowledge search "query" --no-hybrid    # pure vector search (faster, less precise)
knowledge search "query" --rerank       # re-rank with local LLM (slower, most accurate)
knowledge search "query" --no-intent    # skip intent check (always search regardless of topic)
```

### Domain Wiki

The domain wiki is a short markdown file (`domain-data/domain.md`) that describes what your index contains. It is used as an **intent gate** — queries that do not match the domain are rejected before the vector search even runs, so off-topic queries return nothing instead of noisy results.

```bash
knowledge wiki generate --preview   # generate a draft and print it WITHOUT saving (review first)
knowledge wiki generate             # generate and prompt before overwriting existing wiki
knowledge wiki generate --force     # generate and overwrite without prompting
knowledge wiki show                 # print the current wiki
knowledge wiki path                 # print the file path (useful for shell scripting)
knowledge wiki validate             # audit the block list and detect section conflicts
knowledge wiki check "query here"   # dry-run: show intent score and PASS/BLOCK verdict
knowledge wiki check "query" --threshold 0.3   # test with a custom threshold
```

### Avoiding false blocks

The keyword gate extracts terms from the **"Not answerable"** section and blocks any query that contains them. A common mistake is mentioning the **same brand in both sections** — for example, writing "食品等与 Nike 运动无关的品类" in "Not answerable" when Nike is already listed in "Answerable queries". That causes `Nike` to be added to the block list, silently breaking all Nike searches.

Run `knowledge wiki validate` after editing the wiki to catch this before it affects searches. It shows the full block list and flags any terms that appear in both sections:

```
CONFLICTS DETECTED (2 term(s) appear in BOTH sections and will wrongly block legitimate queries):
  jordan  ← remove from 'Not answerable'
  nike    ← remove from 'Not answerable'
```

After generating, you can **edit `domain-data/domain.md` by hand** to improve accuracy. Changes take effect immediately — the embedding cache (`domain-data/domain_emb.json`) is automatically invalidated when the file is saved via the CLI.

**Workflow after a major data change:**
```bash
knowledge index ./new-data --update-wiki   # one step: index + regenerate wiki
# — or separately —
knowledge index ./new-data
knowledge wiki generate
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
{
  "query": "rainy day hiking",
  "top_k": 5,
  "file_type": null,
  "rerank": false,
  "hybrid": true,
  "min_score": 0.45,
  "intent_check": true
}
```

## Configuration

Copy `.env.example` to `.env` and edit as needed. All values can also be set as environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `models/Qwen3-Embedding-0.6B-Q8_0.gguf` | Embedding model — GGUF path or HF repo id |
| `LLM_MODEL` | `models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` | LLM for wiki generation and re-ranking |
| `CHROMA_DIR` | `./chroma` | Vector DB storage |
| `DOMAIN_WIKI_PATH` | `./domain-data/domain.md` | Domain wiki file path |
| `DOMAIN_EMB_PATH` | `./domain-data/domain_emb.json` | Cached wiki embedding (auto-generated) |
| `INTENT_THRESHOLD` | `0.25` | Minimum cosine similarity to pass intent gate |
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

## How Search Works

Search runs in up to four stages:

1. **Intent gate** — embeds the query and compares it to the domain wiki via cosine similarity; queries below the threshold (`INTENT_THRESHOLD`, default 0.25) return nothing immediately. Requires `knowledge wiki generate` to have been run at least once; skipped silently if no wiki exists. Disable per-query with `--no-intent`.
2. **Vector retrieval** — cosine similarity against the ChromaDB embedding index (always on)
3. **Hybrid re-rank** — combines vector rank with BM25 keyword score using Reciprocal Rank Fusion; on by default (`--no-hybrid` to skip)
4. **LLM re-rank** — an on-device LLM scores each passage for relevance; opt-in with `--rerank`

The `min_score` filter (default 0.45) additionally suppresses individual results whose raw cosine similarity is too low, acting as a fallback even when intent checking is off.

### Score interpretation

| Mode | Score meaning |
|------|--------------|
| `--no-hybrid` | Raw cosine similarity (0–1); ~0.65 is a strong match |
| default (hybrid) | RRF-normalised (0–1); top result is always 1.0 |
| `--rerank` | LLM relevance score normalised to 0–1 |

### Domain wiki and intent scores

| Intent score | Meaning |
|---|---|
| < 0.25 | Query is out of scope — returns no results |
| 0.25 – 0.45 | Borderline; in-scope but loosely related |
| > 0.45 | Clearly in scope |

The wiki lives at `domain-data/domain.md` (committed to git so your team can share and edit it). The embedding cache `domain-data/domain_emb.json` is excluded from git and rebuilt automatically when the wiki changes.

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
