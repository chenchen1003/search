# LangChain Loaders + LLM Re-ranker — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all hardcoded domain logic with LangChain loaders (adds HTML support) and a local-LLM re-ranking layer using `Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf`.

**Architecture:** Modify `parser.py` in-place to remove `catalog_filters` coupling and add HTML via `BSHTMLLoader`; add new `knowledge/core/reranker.py` backed by `llama-cpp-python`; wire optional `rerank=True` through `Searcher` → CLI → API.

**Tech Stack:** Python 3.11+, langchain-community, langchain-core, langchain-text-splitters, beautifulsoup4, lxml, llama-cpp-python (already installed), chromadb, fastapi, typer, pytest

---

## Task 1: Install LangChain + HTML dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new packages**

Open `requirements.txt` and replace its contents with:

```
sentence-transformers==3.4.1
llama-cpp-python>=0.3.2
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
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
```

**Step 2: Install**

```bash
pip install langchain-community langchain-core langchain-text-splitters beautifulsoup4 lxml
```

**Step 3: Verify import**

```bash
python -c "from langchain_community.document_loaders import BSHTMLLoader; print('ok')"
```
Expected output: `ok`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add langchain and html parsing dependencies"
```

---

## Task 2: Rewrite parser.py — remove catalog_filters, add HTML

**Files:**
- Modify: `knowledge/core/parser.py`
- Modify: `tests/core/test_parser.py`

**Step 1: Write the new failing tests first**

Replace `tests/core/test_parser.py` entirely:

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


def test_parse_json_array_of_objects(tmp_path):
    import json
    f = tmp_path / "items.json"
    f.write_text(json.dumps([{"id": 1, "name": "foo"}, {"id": 2, "name": "bar"}]))
    chunks = parse_file(f)
    assert len(chunks) == 2
    assert "foo" in chunks[0].text


def test_parse_json_single_object(tmp_path):
    import json
    f = tmp_path / "item.json"
    f.write_text(json.dumps({"title": "test", "value": 42}))
    chunks = parse_file(f)
    assert len(chunks) == 1
    assert "title" in chunks[0].text


def test_parse_html_strips_tags(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body><h1>Hello</h1><p>World content here.</p></body></html>"
    )
    chunks = parse_file(f)
    assert len(chunks) > 0
    full_text = " ".join(c.text for c in chunks)
    assert "Hello" in full_text
    assert "World" in full_text
    assert "<h1>" not in full_text


def test_parse_html_file_type_is_html(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>Test.</p></body></html>")
    chunks = parse_file(f)
    assert all(c.file_type == "html" for c in chunks)


def test_parse_unsupported_extension_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(f)


def test_chunk_has_no_catalog_fields(tmp_path):
    """Ensure no domain-specific fields bleed through."""
    f = tmp_path / "notes.txt"
    f.write_text("Some plain text content.")
    chunks = parse_file(f)
    assert len(chunks) > 0
    chunk = chunks[0]
    assert not hasattr(chunk, "audience") or chunk.extra_metadata.get("audience") is None
```

**Step 2: Run to confirm failures**

```bash
cd /Users/cch102/Desktop/workspace/local-knowledge/search
pytest tests/core/test_parser.py::test_parse_html_strips_tags tests/core/test_parser.py::test_parse_html_file_type_is_html tests/core/test_parser.py::test_chunk_has_no_catalog_fields -v
```
Expected: at least 2 FAIL (html tests) and possibly failures due to catalog_filters still being imported.

**Step 3: Rewrite `knowledge/core/parser.py`**

Replace the entire file:

```python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    extra_metadata: dict = field(default_factory=dict)
    index_text: str | None = None


def parse_file(path: Path) -> list[Chunk]:
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext == "txt":
        return _parse_txt(path)
    if ext == "md":
        return _parse_md(path)
    if ext == "csv":
        return _parse_csv(path)
    if ext == "json":
        return _parse_json(path)
    if ext in ("html", "htm"):
        return _parse_html(path)
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


def _parse_json(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    file_path = str(path)
    chunks: list[Chunk] = []

    if isinstance(data, dict) and "documents" in data and "metadata" in data:
        for i, (doc, meta) in enumerate(zip(data["documents"], data["metadata"])):
            text = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
            extra = meta if isinstance(meta, dict) else {}
            chunks.append(Chunk(
                text=text, file_path=file_path, file_type="json",
                chunk_index=i, extra_metadata=extra,
            ))
        return chunks

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = ", ".join(f"{k}: {v}" for k, v in item.items())
                chunks.append(Chunk(
                    text=text, file_path=file_path, file_type="json",
                    chunk_index=i, extra_metadata=item,
                ))
            else:
                chunks.append(Chunk(
                    text=str(item), file_path=file_path,
                    file_type="json", chunk_index=i,
                ))
        return chunks

    if isinstance(data, dict):
        text = ", ".join(f"{k}: {v}" for k, v in data.items())
        chunks.append(Chunk(
            text=text, file_path=file_path, file_type="json",
            chunk_index=0, extra_metadata=data,
        ))
        return chunks

    chunks.append(Chunk(text=str(data), file_path=file_path, file_type="json", chunk_index=0))
    return chunks


def _parse_html(path: Path, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    from langchain_community.document_loaders import BSHTMLLoader
    loader = BSHTMLLoader(str(path), bs_kwargs={"features": "lxml"})
    docs = loader.load()
    text = " ".join(d.page_content for d in docs)
    # Split into word-based chunks (same strategy as txt)
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(
            text=chunk_text,
            file_path=str(path),
            file_type="html",
            chunk_index=len(chunks),
        ))
        if end == len(words):
            break
        start += chunk_size - overlap
    if not chunks:
        chunks.append(Chunk(text=text.strip(), file_path=str(path), file_type="html", chunk_index=0))
    return chunks
```

**Step 4: Run all parser tests**

```bash
pytest tests/core/test_parser.py -v
```
Expected: all PASSED (9 tests)

**Step 5: Commit**

```bash
git add knowledge/core/parser.py tests/core/test_parser.py
git commit -m "feat: replace catalog-coupled parser with generic loader + HTML support via BSHTMLLoader"
```

---

## Task 3: Delete catalog_filters and clean up index.py

**Files:**
- Delete: `knowledge/core/catalog_filters.py`
- Delete: `tests/core/test_catalog_filters.py`
- Modify: `knowledge/core/index.py`

**Step 1: Remove the files**

```bash
rm knowledge/core/catalog_filters.py tests/core/test_catalog_filters.py
```

**Step 2: Update `knowledge/core/index.py`**

Remove the `enrich_query` import and its usage. Replace the entire file with:

```python
from __future__ import annotations

import json
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

    @staticmethod
    def _safe_meta(value: Any) -> str | int | float | bool:
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, ensure_ascii=False)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        ids = [f"{c.file_path}:{c.chunk_index}" for c in chunks]
        to_embed = [c.index_text if c.index_text else c.text for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        for c in chunks:
            meta: dict[str, Any] = {
                "file_path": c.file_path,
                "file_type": c.file_type,
                "chunk_index": c.chunk_index,
            }
            for k, v in (c.extra_metadata or {}).items():
                meta[k] = self._safe_meta(v)
            metadatas.append(meta)
        vectors = self._embedder.embed(to_embed)
        self._collection.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
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
            hit: dict[str, Any] = {"text": text, "score": round(1 - dist, 4)}
            hit.update(meta)
            output.append(hit)
        return output

    def count(self) -> int:
        return self._collection.count()
```

**Step 3: Run the full test suite to confirm nothing broke**

```bash
pytest -v --ignore=tests/core/test_catalog_filters.py
```
Expected: all tests PASSED (the 12 `test_catalog_filters` tests will no longer exist)

**Step 4: Commit**

```bash
git add knowledge/core/index.py
git rm knowledge/core/catalog_filters.py tests/core/test_catalog_filters.py
git commit -m "refactor: remove catalog_filters domain logic; index.py queries without domain enrichment"
```

---

## Task 4: Create Reranker

**Files:**
- Create: `knowledge/core/reranker.py`
- Create: `tests/core/test_reranker.py`

**Step 1: Write failing tests**

```python
# tests/core/test_reranker.py
import pytest
from unittest.mock import MagicMock, patch
from knowledge.core.reranker import Reranker
from knowledge.core.searcher import SearchResult


def _make_results():
    return [
        SearchResult(text="Cats are mammals", file_path="/a.txt", file_type="txt", chunk_index=0, score=0.9),
        SearchResult(text="Python is a programming language", file_path="/b.txt", file_type="txt", chunk_index=0, score=0.8),
        SearchResult(text="The Eiffel Tower is in Paris", file_path="/c.txt", file_type="txt", chunk_index=0, score=0.7),
    ]


def test_reranker_returns_same_count():
    reranker = Reranker(llm_model_path="/nonexistent.gguf")
    results = _make_results()
    with patch.object(reranker, "_call_llm", return_value=[
        {"index": 0, "score": 3},
        {"index": 1, "score": 9},
        {"index": 2, "score": 1},
    ]):
        reranked = reranker.rerank("programming", results)
    assert len(reranked) == 3


def test_reranker_sorts_by_llm_score():
    reranker = Reranker(llm_model_path="/nonexistent.gguf")
    results = _make_results()
    with patch.object(reranker, "_call_llm", return_value=[
        {"index": 0, "score": 3},
        {"index": 1, "score": 9},
        {"index": 2, "score": 1},
    ]):
        reranked = reranker.rerank("programming", results)
    assert reranked[0].text == "Python is a programming language"
    assert reranked[-1].text == "The Eiffel Tower is in Paris"


def test_reranker_fallback_on_llm_error():
    """When LLM fails, returns results in original order."""
    reranker = Reranker(llm_model_path="/nonexistent.gguf")
    results = _make_results()
    with patch.object(reranker, "_call_llm", side_effect=Exception("model unavailable")):
        reranked = reranker.rerank("any query", results)
    assert [r.text for r in reranked] == [r.text for r in results]


def test_reranker_updates_score_field():
    reranker = Reranker(llm_model_path="/nonexistent.gguf")
    results = _make_results()
    with patch.object(reranker, "_call_llm", return_value=[
        {"index": 0, "score": 5},
        {"index": 1, "score": 8},
        {"index": 2, "score": 2},
    ]):
        reranked = reranker.rerank("query", results)
    # scores should now be normalised 0-1 (llm_score / 10)
    assert all(0.0 <= r.score <= 1.0 for r in reranked)


def test_reranker_model_missing_falls_back():
    """Reranker constructed with missing model path still returns results."""
    reranker = Reranker(llm_model_path="/this/does/not/exist.gguf")
    results = _make_results()
    reranked = reranker.rerank("cats", results)
    assert len(reranked) == 3
```

**Step 2: Run to confirm failures**

```bash
pytest tests/core/test_reranker.py -v
```
Expected: FAIL with `ImportError` or `ModuleNotFoundError`

**Step 3: Implement `knowledge/core/reranker.py`**

```python
from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)

_SCORE_PROMPT = """\
You are a search relevance judge. Given a query and a list of passages, score each passage 0-10 for how relevant it is to the query.

Query: {query}

Passages:
{passages}

Reply ONLY with a JSON array, one object per passage: [{{"index": 0, "score": 7}}, ...]
Do not explain. Output JSON only."""


class Reranker:
    def __init__(self, llm_model_path: str) -> None:
        self._model_path = llm_model_path
        self._llm = None

    def _load_llm(self):
        if self._llm is not None:
            return
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM model not found: {self._model_path}")
        from llama_cpp import Llama
        import os
        devnull = open(os.devnull, "w")
        self._llm = Llama(
            model_path=str(path),
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

    def _call_llm(self, prompt: str) -> list[dict]:
        self._load_llm()
        response = self._llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
            stop=["```"],
        )
        raw = response["choices"][0]["text"].strip()
        # Extract JSON array from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON array in LLM response: {raw!r}")
        return json.loads(match.group())

    def rerank(self, query: str, results: list) -> list:
        if not results:
            return results
        passages = "\n".join(
            f"[{i}] {r.text[:300]}" for i, r in enumerate(results)
        )
        prompt = _SCORE_PROMPT.format(query=query, passages=passages)
        try:
            scores_raw = self._call_llm(prompt)
            score_map = {item["index"]: item["score"] for item in scores_raw}
        except Exception as exc:
            logger.warning("Reranker LLM call failed, using original order: %s", exc)
            return results

        scored = []
        for i, r in enumerate(results):
            llm_score = score_map.get(i, 0)
            scored.append((llm_score / 10.0, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [replace(r, score=round(s, 4)) for s, r in scored]
```

**Step 4: Run tests**

```bash
pytest tests/core/test_reranker.py -v
```
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add knowledge/core/reranker.py tests/core/test_reranker.py
git commit -m "feat: LLM re-ranker using Qwen2.5-Coder GGUF via llama-cpp-python"
```

---

## Task 5: Wire Reranker into Searcher

**Files:**
- Modify: `knowledge/core/searcher.py`
- Modify: `knowledge/config.py`

**Step 1: Write failing test**

Add this test to `tests/core/test_searcher.py`:

```python
def test_search_with_rerank_returns_results(searcher, tmp_path):
    f = tmp_path / "facts.txt"
    f.write_text("Python is a programming language used for data science.")
    searcher.index_path(f)
    # rerank=True but model path won't exist in test env → falls back gracefully
    results = searcher.search("data science language", top_k=1, rerank=True)
    assert len(results) == 1
    assert isinstance(results[0].score, float)
```

**Step 2: Run to confirm failure**

```bash
pytest tests/core/test_searcher.py::test_search_with_rerank_returns_results -v
```
Expected: FAIL — `search()` does not accept `rerank` argument yet.

**Step 3: Update `knowledge/config.py`**

Add `llm_model` setting. Replace the Settings class body:

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    return Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    chroma_dir: Path = Path.home() / ".local" / "share" / "knowledge" / "chroma"
    embed_model: str = "Qwen/Qwen3-Embedding"
    llm_model: str = str(_repo_root() / "models" / "Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf")
    chunk_size: int = 400
    chunk_overlap: int = 50
    api_port: int = 8000


settings = Settings()
```

**Step 4: Update `knowledge/core/searcher.py`**

Replace the entire file:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from knowledge.core.index import VectorIndex
from knowledge.core.parser import parse_file
from knowledge.config import settings


@dataclass
class SearchResult:
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    score: float
    extra_metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".html", ".htm"}


class Searcher:
    def __init__(self, chroma_dir: Path, embed_model: str, llm_model: str | None = None) -> None:
        self._index = VectorIndex(chroma_dir=chroma_dir, embed_model=embed_model)
        self._llm_model = llm_model or settings.llm_model

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
        rerank: bool = False,
    ) -> list[SearchResult]:
        fetch_k = top_k * 2 if rerank else top_k
        hits = self._index.query(query, top_k=fetch_k, file_type=file_type)
        fixed_keys = {"text", "file_path", "file_type", "chunk_index", "score"}
        results = [
            SearchResult(
                text=h["text"],
                file_path=h["file_path"],
                file_type=h["file_type"],
                chunk_index=h["chunk_index"],
                score=h["score"],
                extra_metadata={k: v for k, v in h.items() if k not in fixed_keys},
            )
            for h in hits
        ]
        if rerank and results:
            from knowledge.core.reranker import Reranker
            reranker = Reranker(llm_model_path=self._llm_model)
            results = reranker.rerank(query, results)
        return results[:top_k]

    def delete(self, path: Path) -> int:
        return self._index.delete(str(path))
```

**Step 5: Run searcher tests**

```bash
pytest tests/core/test_searcher.py -v
```
Expected: all PASSED

**Step 6: Commit**

```bash
git add knowledge/core/searcher.py knowledge/config.py
git commit -m "feat: wire Reranker into Searcher.search(rerank=True); add llm_model config"
```

---

## Task 6: Add --rerank flag to CLI and API

**Files:**
- Modify: `knowledge/cli/main.py`
- Modify: `knowledge/api/server.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_api.py`

**Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
def test_search_command_rerank_flag(tmp_path):
    chroma = tmp_path / "chroma"
    f = tmp_path / "notes.txt"
    f.write_text("Penguins live in Antarctica.")
    runner.invoke(app, ["index", str(f), "--chroma-dir", str(chroma),
                        "--model", "sentence-transformers/all-MiniLM-L6-v2"])
    result = runner.invoke(app, ["search", "cold birds", "--chroma-dir", str(chroma),
                                 "--model", "sentence-transformers/all-MiniLM-L6-v2",
                                 "--rerank"])
    # rerank falls back gracefully when model is missing
    assert result.exit_code == 0
```

Add to `tests/test_api.py`:

```python
def test_search_endpoint_rerank_flag(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The ocean is vast and deep.")
    client.post("/index", json={"path": str(f)})
    resp = client.post("/search", json={"query": "sea water", "top_k": 1, "rerank": True})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
```

**Step 2: Run to confirm failure**

```bash
pytest tests/test_cli.py::test_search_command_rerank_flag tests/test_api.py::test_search_endpoint_rerank_flag -v
```
Expected: FAIL

**Step 3: Update `knowledge/cli/main.py`**

Replace entire file:

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
    file_type: Optional[str] = typer.Option(None, "--type", help="Filter by file type"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Re-rank results with local LLM"),
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
) -> None:
    """Semantic search over indexed files."""
    searcher = _make_searcher(chroma_dir, model)
    results = searcher.search(query, top_k=top_k, file_type=file_type, rerank=rerank)
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

**Step 4: Update `knowledge/api/server.py`**

Replace entire file:

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
    rerank: bool = False


class SearchResult(BaseModel):
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    score: float
    extra_metadata: dict = {}


class DeleteRequest(BaseModel):
    file_path: str


class DeleteResponse(BaseModel):
    deleted: int


def create_app(chroma_dir: Path, embed_model: str) -> FastAPI:
    searcher = Searcher(chroma_dir=chroma_dir, embed_model=embed_model)
    api = FastAPI(title="Local Knowledge Search", version="0.2.0")

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
        results = searcher.search(
            req.query, top_k=req.top_k,
            file_type=req.file_type, rerank=req.rerank,
        )
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

**Step 5: Run all tests**

```bash
pytest -v
```
Expected: all PASSED

**Step 6: Commit**

```bash
git add knowledge/cli/main.py knowledge/api/server.py tests/test_cli.py tests/test_api.py
git commit -m "feat: add --rerank/--no-rerank flag to CLI and rerank field to API search endpoint"
```

---

## Task 7: Update README and final verification

**Files:**
- Modify: `README.md`

**Step 1: Run full test suite one final time**

```bash
pytest -v
```
Expected: all PASSED, no failures.

**Step 2: Manual smoke test**

```bash
knowledge --help
```
Expected: shows `index`, `search`, `delete` with `--rerank` documented under `search`.

**Step 3: Update README**

Open `README.md` and replace the CLI Usage section with:

```markdown
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
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for HTML support, --rerank flag, and GGUF model config"
```
