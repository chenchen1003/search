import pytest
from pathlib import Path
from conftest import requires_embed_model, EMBED_MODEL
from knowledge.core.index import VectorIndex
from knowledge.core.parser import Chunk


@pytest.fixture
def index(tmp_path):
    return VectorIndex(chroma_dir=tmp_path, embed_model=EMBED_MODEL)


@requires_embed_model
def test_add_and_count(index):
    chunks = [
        Chunk(text="The sky is blue", file_path="/notes/a.txt", file_type="txt", chunk_index=0),
        Chunk(text="Cats are mammals", file_path="/notes/a.txt", file_type="txt", chunk_index=1),
    ]
    index.add(chunks)
    assert index.count() == 2


@requires_embed_model
def test_add_is_idempotent(index):
    chunk = Chunk(text="hello", file_path="/f.txt", file_type="txt", chunk_index=0)
    index.add([chunk])
    index.add([chunk])
    assert index.count() == 1


@requires_embed_model
def test_delete_by_file_path(index):
    chunks = [
        Chunk(text="A", file_path="/a.txt", file_type="txt", chunk_index=0),
        Chunk(text="B", file_path="/b.txt", file_type="txt", chunk_index=0),
    ]
    index.add(chunks)
    deleted = index.delete("/a.txt")
    assert deleted == 1
    assert index.count() == 1


@requires_embed_model
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


@requires_embed_model
def test_query_filter_by_file_type(index):
    chunks = [
        Chunk(text="markdown content", file_path="/doc.md", file_type="md", chunk_index=0),
        Chunk(text="csv row data", file_path="/data.csv", file_type="csv", chunk_index=0),
    ]
    index.add(chunks)
    results = index.query("content", top_k=10, file_type="md")
    assert all(r["file_type"] == "md" for r in results)
