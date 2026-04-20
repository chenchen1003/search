# Local Knowledge Search

Fully local semantic search over Markdown, CSV, JSON, plain-text, and **HTML** files.
No cloud API. By default embeddings use a **project-local GGUF** at
`models/Qwen3-Embedding-0.6B-Q8_0.gguf` (resolved relative to the `search` repo root, so the CLI works from any working directory). You can switch to a HuggingFace SentenceTransformer id via `EMBED_MODEL` in `.env`.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # edit paths/model if needed
```

### Option A — GGUF + llama.cpp (recommended if HuggingFace / XetHub is blocked)

1. Download one ``*.gguf`` file from [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) (e.g. Q8_0 — pick a file that fits your RAM).

2. Install bindings (Apple Silicon — Metal acceleration):

   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

   On other platforms, plain ``pip install llama-cpp-python`` is usually enough.

3. Point the app at the file:

   ```bash
   export EMBED_MODEL=/absolute/path/to/your-model.gguf
   # or add the same line to .env
   knowledge index ./raw-data/index_meta.json
   ```

**Important:** If you previously indexed with the PyTorch / SentenceTransformer model, delete your Chroma folder (or use a new ``CHROMA_DIR``) before switching — vector dimensions must match.

### Option B — PyTorch / SentenceTransformer (HuggingFace)

The model (~1.2 GB) must be downloaded once before indexing or searching.
If your network uses a corporate SSL proxy, export your system CA bundle first:

```bash
security export -t certs -f pemseq -k /Library/Keychains/System.keychain \
    -o /tmp/system_certs.pem
security export -t certs -f pemseq \
    -k /System/Library/Keychains/SystemRootCertificates.keychain \
    >> /tmp/system_certs.pem

REQUESTS_CA_BUNDLE=/tmp/system_certs.pem \
SSL_CERT_FILE=/tmp/system_certs.pem \
HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-0.6B')
"
```

Then either keep ``EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B`` (default) or pass ``--model`` on the CLI.

### Environment tuning (GGUF only)

| Variable | Default | Meaning |
|----------|---------|---------|
| ``KNOWLEDGE_LLAMA_N_CTX`` | ``8192`` | Context length (raise if long documents are truncated) |
| ``KNOWLEDGE_LLAMA_N_GPU_LAYERS`` | ``-1`` | GPU layers (‑1 = all on Metal/CUDA; ``0`` = CPU only) |
| ``KNOWLEDGE_LLAMA_VERBOSE`` | (off) | Set to ``1`` to show llama.cpp stderr (e.g. ``init: embeddings required…``); default hides that harmless spam |

## CLI Usage

```bash
knowledge index ./my-notes              # index a folder (md, txt, csv, json, html)
knowledge search "query here"           # semantic search
knowledge search "query" --top-k 10 --type md
knowledge search "query" --rerank       # re-rank results with local LLM (slower, smarter)
knowledge delete ./my-notes/old.md      # remove from index
```

## API Server

```bash
python -m knowledge.api
# Server runs at http://localhost:8000
```

Endpoints: `POST /index`, `POST /search`, `DELETE /document`, `GET /health`

`POST /search` accepts `"rerank": true` to enable LLM re-ranking.

## Models

Place GGUF models in the `models/` folder:
- `Qwen3-Embedding-0.6B-Q8_0.gguf` — embedding model (used by default)
- `Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` — re-ranking LLM (used with `--rerank`)

Override via `.env`:
```
EMBED_MODEL=./models/Qwen3-Embedding-0.6B-Q8_0.gguf
LLM_MODEL=./models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf
```

## Configuration

All settings can be overridden via `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `models/Qwen3-Embedding-0.6B-Q8_0.gguf` | Embedding model (GGUF path or HF repo id) |
| `LLM_MODEL` | `models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf` | Re-ranking LLM (GGUF path) |
| `CHROMA_DIR` | `./chroma` (inside project folder) | ChromaDB storage path |
| `CHUNK_SIZE` | `400` | Words per chunk (txt/html files) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `API_PORT` | `8000` | REST API port |

## Supported File Types

| Extension | Chunking strategy |
|-----------|------------------|
| `.txt` | Fixed-size word windows with overlap |
| `.md` | Split on `#` headings |
| `.csv` | One chunk per row |
| `.json` | One chunk per array item or object |
| `.html` / `.htm` | Tag-stripped, then fixed-size word windows |

## Running Tests

```bash
pytest -v
# Parser tests run immediately.
# Embedding tests skip until Qwen3-Embedding-0.6B is downloaded.
```
