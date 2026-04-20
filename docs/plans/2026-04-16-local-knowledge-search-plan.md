# Local Knowledge Search Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully local semantic search engine for Markdown, CSV, and plain text files with a REST API and CLI.

**Architecture:** A single Python package (`knowledge`) with a shared `core/` library (parser, embedder, index, searcher) consumed by both a `FastAPI` API server and a `typer` CLI. ChromaDB stores vectors on disk; Qwen3-Embedding runs locally via `sentence-transformers`.

**Tech Stack:** Python 3.11+, sentence-transformers, Qwen/Qwen3-Embedding, chromadb, fastapi, uvicorn, typer, markdown-it-py, pydantic-settings, pytest

---

## Task 1: Project Scaffolding

**Files:**
- Create: `knowledge/__init__.py`
- Create: `knowledge/core/__init__.py`
- Create: `knowledge/api/__init__.py`
- Create: `knowledge/cli/__init__.py`
- Create: `knowledge/config.py`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`

**Step 1: Create the directory structure**

```bash
mkdir -p knowledge/core knowledge/api knowledge/cli tests/core
touch knowledge/__init__.py knowledge/core/__init__.py knowledge/api/__init__.py knowledge/cli/__init__.py
touch tests/__init__.py tests/core/__init__.py
```

**Step 2: Create `requirements.txt`**

```
sentence-transformers==3.4.1
chromadb==0.6.3
fastapi==0.115.6
uvicorn[standard]==0.34.0
typer==0.15.1
markdown-it-py==3.0.0
pydantic-settings==2.7.1
python-dotenv==1.0.1
pytest==8.3.4
pytest-asyncio==0.25.0
httpx==0.28.1
```

**Step 3: Create `knowledge/config.py`**

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    chroma_dir: Path = Path.home() / ".local" / "share" / "knowledge" / "chroma"
    embed_model: str = "Qwen/Qwen3-Embedding"
    chunk_size: int = 400
    chunk_overlap: int = 50
    api_port: int = 8000


settings = Settings()
```

**Step 4: Create `.env.example`**

```
CHROMA_DIR=~/.local/share/knowledge/chroma
EMBED_MODEL=Qwen/Qwen3-Embedding
CHUNK_SIZE=400
CHUNK_OVERLAP=50
API_PORT=8000
```

**Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Verify imports work**

```bash
python -c "from knowledge.config import settings; print(settings.embed_model)"
```
Expected output: `Qwen/Qwen3-Embedding`

**Step 7: Commit**

```bash
git add knowledge/ tests/ requirements.txt .env.example
git commit -m "feat: project scaffold with config"
```

---

## Task 2: File Parser

**Files:**
- Create: `knowledge/core/parser.py`
- Create: `tests/core/test_parser.py`

**Step 1: Write failing tests**

```python
# tests/core/test_parser.py
import textwrap
import pytest
from knowledge.core.parser import parse_file, Chunk


def test_parse_txt_returns_chunks(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Hello world. " * 100)
    chunks = parse_file(f)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.file_path == str(f) for c in chunks)
    assert all(c.file_type == "txt" for c in chunks)


def test_parse_txt_chunk_index_sequential(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("word " * 500)
    chunks = parse_file(f)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_parse_md_splits_by_heading(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Section One\nContent one.\n\n# Section Two\nContent two.\n")
    chunks = parse_file(f)
    assert len(chunks) == 2
    assert "Section One" in chunks[0].text
    assert "Section Two" in chunks[1].text


def test_parse_csv_one_chunk_per_row(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25\n")
    chunks = parse_file(f)
    assert len(chunks) == 2
    assert "name: Alice" in chunks[0].text
    assert "age: 30" in chunks[0].text


def test_parse_unsupported_extension_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(f)
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_parser.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Implement `knowledge/core/parser.py`**

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    file_path: str
    file_type: str
    chunk_index: int


def parse_file(path: Path) -> list[Chunk]:
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext == "txt":
        return _parse_txt(path)
    if ext == "md":
        return _parse_md(path)
    if ext == "csv":
        return _parse_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _parse_txt(path: Path, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(
            text=chunk_text,
            file_path=str(path),
            file_type="txt",
            chunk_index=len(chunks),
        ))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _parse_md(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#") and current:
            sections.append("".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current).strip())
    sections = [s for s in sections if s]
    if not sections:
        sections = [text.strip()]
    return [
        Chunk(text=s, file_path=str(path), file_type="md", chunk_index=i)
        for i, s in enumerate(sections)
    ]


def _parse_csv(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = ", ".join(f"{k}: {v}" for k, v in row.items())
            chunks.append(Chunk(text=text, file_path=str(path), file_type="csv", chunk_index=i))
    return chunks
```

**Step 4: Run tests to confirm they pass**

```bash
pytest tests/core/test_parser.py -v
```
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add knowledge/core/parser.py tests/core/test_parser.py
git commit -m "feat: file parser for md, csv, txt"
```

---

## Task 3: Embedder

**Files:**
- Create: `knowledge/core/embedder.py`
- Create: `tests/core/test_embedder.py`

> **Note:** Tests use a tiny fast model (`all-MiniLM-L6-v2`) to avoid downloading Qwen3 during CI. Production config uses `Qwen/Qwen3-Embedding`.

**Step 1: Write failing tests**

```python
# tests/core/test_embedder.py
import pytest
from knowledge.core.embedder import Embedder

FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_embed_returns_list_of_floats():
    embedder = Embedder(model_name=FAST_MODEL)
    vectors = embedder.embed(["hello world"])
    assert len(vectors) == 1
    assert isinstance(vectors[0], list)
    assert all(isinstance(v, float) for v in vectors[0])


def test_embed_batch_returns_correct_count():
    embedder = Embedder(model_name=FAST_MODEL)
    texts = ["first", "second", "third"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3


def test_embed_same_text_produces_same_vector():
    embedder = Embedder(model_name=FAST_MODEL)
    v1 = embedder.embed(["test sentence"])[0]
    v2 = embedder.embed(["test sentence"])[0]
    assert v1 == v2


def test_embed_different_texts_produce_different_vectors():
    embedder = Embedder(model_name=FAST_MODEL)
    v1 = embedder.embed(["cat"])[0]
    v2 = embedder.embed(["quantum physics"])[0]
    assert v1 != v2
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_embedder.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement `knowledge/core/embedder.py`**

```python
from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
```

**Step 4: Run tests to confirm they pass**

```bash
pytest tests/core/test_embedder.py -v
```
Expected: 4 PASSED (first run downloads `all-MiniLM-L6-v2`, ~90MB)

**Step 5: Commit**

```bash
git add knowledge/core/embedder.py tests/core/test_embedder.py
git commit -m "feat: embedder wrapping sentence-transformers"
```

---

## Task 4: Vector Index (ChromaDB)

**Files:**
- Create: `knowledge/core/index.py`
- Create: `tests/core/test_index.py`

**Step 1: Write failing tests**

```python
# tests/core/test_index.py
import pytest
from pathlib import Path
from knowledge.core.index import VectorIndex
from knowledge.core.parser import Chunk

FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def index(tmp_path):
    return VectorIndex(chroma_dir=tmp_path, embed_model=FAST_MODEL)


def test_add_and_count(index):
    chunks = [
        Chunk(text="The sky is blue", file_path="/notes/a.txt", file_type="txt", chunk_index=0),
        Chunk(text="Cats are mammals", file_path="/notes/a.txt", file_type="txt", chunk_index=1),
    ]
    index.add(chunks)
    assert index.count() == 2


def test_add_is_idempotent(index):
    chunk = Chunk(text="hello", file_path="/f.txt", file_type="txt", chunk_index=0)
    index.add([chunk])
    index.add([chunk])
    assert index.count() == 1


def test_delete_by_file_path(index):
    chunks = [
        Chunk(text="A", file_path="/a.txt", file_type="txt", chunk_index=0),
        Chunk(text="B", file_path="/b.txt", file_type="txt", chunk_index=0),
    ]
    index.add(chunks)
    deleted = index.delete("/a.txt")
    assert deleted == 1
    assert index.count() == 1


def test_query_returns_results(index):
    chunks = [
        Chunk(text="Python is a programming language", file_path="/p.md", file_type="md", chunk_index=0),
        Chunk(text="The Eiffel Tower is in Paris", file_path="/p.md", file_type="md", chunk_index=1),
    ]
    index.add(chunks)
    results = index.query("coding languages", top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "Python is a programming language"
    assert "score" in results[0]
    assert "file_path" in results[0]


def test_query_filter_by_file_type(index):
    chunks = [
        Chunk(text="markdown content", file_path="/doc.md", file_type="md", chunk_index=0),
        Chunk(text="csv row data", file_path="/data.csv", file_type="csv", chunk_index=0),
    ]
    index.add(chunks)
    results = index.query("content", top_k=10, file_type="md")
    assert all(r["file_type"] == "md" for r in results)
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_index.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement `knowledge/core/index.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from knowledge.core.embedder import Embedder
from knowledge.core.parser import Chunk

COLLECTION_NAME = "knowledge"


class VectorIndex:
    def __init__(self, chroma_dir: Path, embed_model: str) -> None:
        chroma_dir = Path(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = Embedder(embed_model)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        ids = [f"{c.file_path}:{c.chunk_index}" for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "file_type": c.file_type,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        vectors = self._embedder.embed(texts)
        self._collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        return len(chunks)

    def delete(self, file_path: str) -> int:
        results = self._collection.get(where={"file_path": file_path})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        file_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where = {"file_type": file_type} if file_type else None
        vector = self._embedder.embed([query_text])[0]
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": text,
                "file_path": meta["file_path"],
                "file_type": meta["file_type"],
                "chunk_index": meta["chunk_index"],
                "score": round(1 - dist, 4),
            })
        return output

    def count(self) -> int:
        return self._collection.count()
```

**Step 4: Run tests to confirm they pass**

```bash
pytest tests/core/test_index.py -v
```
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add knowledge/core/index.py tests/core/test_index.py
git commit -m "feat: vector index backed by ChromaDB"
```

---

## Task 5: Searcher (Public Interface)

**Files:**
- Create: `knowledge/core/searcher.py`
- Create: `tests/core/test_searcher.py`

**Step 1: Write failing tests**

```python
# tests/core/test_searcher.py
import pytest
from pathlib import Path
from knowledge.core.searcher import Searcher, SearchResult

FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def searcher(tmp_path):
    return Searcher(chroma_dir=tmp_path, embed_model=FAST_MODEL)


def test_index_file_and_search(searcher, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Machine learning is about training models on data.")
    indexed = searcher.index_path(f)
    assert indexed > 0
    results = searcher.search("neural networks and training", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].file_path == str(f)


def test_index_directory(searcher, tmp_path):
    (tmp_path / "a.txt").write_text("The sun rises in the east.")
    (tmp_path / "b.txt").write_text("Water boils at 100 degrees.")
    total = searcher.index_path(tmp_path)
    assert total == 2


def test_delete_document(searcher, tmp_path):
    f = tmp_path / "del.txt"
    f.write_text("This will be deleted.")
    searcher.index_path(f)
    deleted = searcher.delete(f)
    assert deleted >= 1
    results = searcher.search("deleted content", top_k=5)
    assert all(r.file_path != str(f) for r in results)


def test_search_result_has_required_fields(searcher, tmp_path):
    f = tmp_path / "check.md"
    f.write_text("# Topic\nSome content here.")
    searcher.index_path(f)
    results = searcher.search("topic content")
    assert len(results) > 0
    r = results[0]
    assert hasattr(r, "text")
    assert hasattr(r, "file_path")
    assert hasattr(r, "score")
    assert hasattr(r, "chunk_index")
    assert 0.0 <= r.score <= 1.0
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_searcher.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement `knowledge/core/searcher.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge.core.index import VectorIndex
from knowledge.core.parser import parse_file


@dataclass
class SearchResult:
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    score: float


SUPPORTED_EXTENSIONS = {".md", ".txt", ".csv"}


class Searcher:
    def __init__(self, chroma_dir: Path, embed_model: str) -> None:
        self._index = VectorIndex(chroma_dir=chroma_dir, embed_model=embed_model)

    def index_path(self, path: Path) -> int:
        path = Path(path)
        if path.is_dir():
            total = 0
            for f in path.rglob("*"):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    total += self._index_file(f)
            return total
        return self._index_file(path)

    def _index_file(self, path: Path) -> int:
        chunks = parse_file(path)
        return self._index.add(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_type: str | None = None,
    ) -> list[SearchResult]:
        hits = self._index.query(query, top_k=top_k, file_type=file_type)
        return [
            SearchResult(
                text=h["text"],
                file_path=h["file_path"],
                file_type=h["file_type"],
                chunk_index=h["chunk_index"],
                score=h["score"],
            )
            for h in hits
        ]

    def delete(self, path: Path) -> int:
        return self._index.delete(str(path))
```

**Step 4: Run tests to confirm they pass**

```bash
pytest tests/core/test_searcher.py -v
```
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add knowledge/core/searcher.py tests/core/test_searcher.py
git commit -m "feat: searcher public interface over vector index"
```

---

## Task 6: CLI

**Files:**
- Create: `knowledge/cli/main.py`
- Create: `tests/test_cli.py`
- Modify: `pyproject.toml` or `setup.py` (add entry point)

**Step 1: Write failing tests**

```python
# tests/test_cli.py
import pytest
from typer.testing import CliRunner
from knowledge.cli.main import app

runner = CliRunner()


def test_index_command_success(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Some text content here.")
    result = runner.invoke(app, ["index", str(f), "--chroma-dir", str(tmp_path / "chroma"),
                                 "--model", "sentence-transformers/all-MiniLM-L6-v2"])
    assert result.exit_code == 0
    assert "Indexed" in result.output


def test_search_command_success(tmp_path):
    chroma = tmp_path / "chroma"
    f = tmp_path / "notes.txt"
    f.write_text("Penguins live in Antarctica.")
    runner.invoke(app, ["index", str(f), "--chroma-dir", str(chroma),
                        "--model", "sentence-transformers/all-MiniLM-L6-v2"])
    result = runner.invoke(app, ["search", "cold weather birds", "--chroma-dir", str(chroma),
                                 "--model", "sentence-transformers/all-MiniLM-L6-v2"])
    assert result.exit_code == 0
    assert "Antarctica" in result.output or len(result.output) > 0


def test_delete_command_success(tmp_path):
    chroma = tmp_path / "chroma"
    f = tmp_path / "del.txt"
    f.write_text("Content to remove.")
    runner.invoke(app, ["index", str(f), "--chroma-dir", str(chroma),
                        "--model", "sentence-transformers/all-MiniLM-L6-v2"])
    result = runner.invoke(app, ["delete", str(f), "--chroma-dir", str(chroma),
                                 "--model", "sentence-transformers/all-MiniLM-L6-v2"])
    assert result.exit_code == 0
    assert "Deleted" in result.output
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cli.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement `knowledge/cli/main.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from knowledge.config import settings
from knowledge.core.searcher import Searcher

app = typer.Typer(help="Local knowledge semantic search engine.")


def _make_searcher(chroma_dir: Path, model: str) -> Searcher:
    return Searcher(chroma_dir=chroma_dir, embed_model=model)


@app.command()
def index(
    path: Path = typer.Argument(..., help="File or directory to index"),
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
) -> None:
    """Index a file or directory."""
    searcher = _make_searcher(chroma_dir, model)
    total = searcher.index_path(path)
    typer.echo(f"Indexed {total} chunk(s) from {path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results"),
    file_type: Optional[str] = typer.Option(None, "--type", help="Filter by file type (md/csv/txt)"),
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
) -> None:
    """Semantic search over indexed files."""
    searcher = _make_searcher(chroma_dir, model)
    results = searcher.search(query, top_k=top_k, file_type=file_type)
    if not results:
        typer.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        typer.echo(f"\n[{i}] score={r.score:.3f}  {r.file_path}  (chunk {r.chunk_index})")
        typer.echo(f"    {r.text[:200]}")


@app.command()
def delete(
    path: Path = typer.Argument(..., help="File path to remove from the index"),
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
) -> None:
    """Remove a file's chunks from the index."""
    searcher = _make_searcher(chroma_dir, model)
    deleted = searcher.delete(path)
    typer.echo(f"Deleted {deleted} chunk(s) for {path}")


if __name__ == "__main__":
    app()
```

**Step 4: Add entry point — create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "knowledge"
version = "0.1.0"
requires-python = ">=3.11"

[project.scripts]
knowledge = "knowledge.cli.main:app"
```

Install in editable mode:

```bash
pip install -e .
```

**Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_cli.py -v
```
Expected: 3 PASSED

**Step 6: Verify CLI works manually**

```bash
knowledge --help
```
Expected: shows `index`, `search`, `delete` commands

**Step 7: Commit**

```bash
git add knowledge/cli/main.py tests/test_cli.py pyproject.toml
git commit -m "feat: typer CLI with index, search, delete commands"
```

---

## Task 7: REST API

**Files:**
- Create: `knowledge/api/server.py`
- Create: `tests/test_api.py`

**Step 1: Write failing tests**

```python
# tests/test_api.py
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("EMBED_MODEL", FAST_MODEL)
    # Re-import after env patch so settings picks up new values
    import importlib
    import knowledge.config as cfg
    importlib.reload(cfg)
    from knowledge.api.server import create_app
    return TestClient(create_app(chroma_dir=tmp_path / "chroma", embed_model=FAST_MODEL))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_endpoint(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The ocean is vast and deep.")
    resp = client.post("/index", json={"path": str(f)})
    assert resp.status_code == 200
    assert resp.json()["indexed"] >= 1


def test_search_endpoint(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The ocean is vast and deep.")
    client.post("/index", json={"path": str(f)})
    resp = client.post("/search", json={"query": "sea water", "top_k": 1})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert "text" in results[0]
    assert "score" in results[0]
    assert "file_path" in results[0]


def test_delete_endpoint(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Temporary content.")
    client.post("/index", json={"path": str(f)})
    resp = client.delete("/document", json={"file_path": str(f)})
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 1


def test_index_nonexistent_path_returns_404(client):
    resp = client.post("/index", json={"path": "/nonexistent/path"})
    assert resp.status_code == 404
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_api.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Implement `knowledge/api/server.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from knowledge.core.searcher import Searcher


class IndexRequest(BaseModel):
    path: str


class IndexResponse(BaseModel):
    indexed: int
    skipped: int = 0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    file_type: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    score: float


class DeleteRequest(BaseModel):
    file_path: str


class DeleteResponse(BaseModel):
    deleted: int


def create_app(chroma_dir: Path, embed_model: str) -> FastAPI:
    searcher = Searcher(chroma_dir=chroma_dir, embed_model=embed_model)
    api = FastAPI(title="Local Knowledge Search", version="0.1.0")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/index", response_model=IndexResponse)
    def index(req: IndexRequest):
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
        indexed = searcher.index_path(path)
        return IndexResponse(indexed=indexed)

    @api.post("/search", response_model=list[SearchResult])
    def search(req: SearchRequest):
        results = searcher.search(req.query, top_k=req.top_k, file_type=req.file_type)
        return [SearchResult(**vars(r)) for r in results]

    @api.delete("/document", response_model=DeleteResponse)
    def delete_document(req: DeleteRequest):
        deleted = searcher.delete(Path(req.file_path))
        return DeleteResponse(deleted=deleted)

    return api


def run(chroma_dir: Path, embed_model: str, port: int = 8000) -> None:
    import uvicorn
    app = create_app(chroma_dir=chroma_dir, embed_model=embed_model)
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**Step 4: Add `knowledge/api/__main__.py` so the server is runnable**

```python
# knowledge/api/__main__.py
from knowledge.config import settings
from knowledge.api.server import run

run(chroma_dir=settings.chroma_dir, embed_model=settings.embed_model, port=settings.api_port)
```

**Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_api.py -v
```
Expected: 5 PASSED

**Step 6: Run the full test suite**

```bash
pytest -v
```
Expected: All tests PASSED

**Step 7: Commit**

```bash
git add knowledge/api/ tests/test_api.py
git commit -m "feat: FastAPI server with index, search, delete endpoints"
```

---

## Task 8: Final Wiring & README

**Files:**
- Create: `README.md`

**Step 1: Create `README.md`**

```markdown
# Local Knowledge Search

Fully local semantic search over Markdown, CSV, and plain-text files.
No cloud API. Runs entirely on your machine.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # edit paths/model if needed
```

## CLI Usage

```bash
knowledge index ./my-notes          # index a folder
knowledge search "query here"       # search
knowledge search "query" --top-k 10 --type md
knowledge delete ./my-notes/old.md  # remove from index
```

## API Server

```bash
python -m knowledge.api
# Server runs at http://localhost:8000
```

Endpoints: `POST /index`, `POST /search`, `DELETE /document`, `GET /health`

## First Run

The embedding model (`Qwen/Qwen3-Embedding`) is downloaded automatically on first use (~1–3 GB).
```

**Step 2: Run full test suite one final time**

```bash
pytest -v
```
Expected: All PASSED

**Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```
