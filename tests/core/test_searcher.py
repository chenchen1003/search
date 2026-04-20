import pytest
from pathlib import Path
from conftest import requires_embed_model, EMBED_MODEL
from knowledge.core.searcher import Searcher, SearchResult


@pytest.fixture
def searcher(tmp_path):
    return Searcher(chroma_dir=tmp_path, embed_model=EMBED_MODEL)


@requires_embed_model
def test_index_file_and_search(searcher, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Machine learning is about training models on data.")
    indexed = searcher.index_path(f)
    assert indexed > 0
    results = searcher.search("neural networks and training", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].file_path == str(f)


@requires_embed_model
def test_index_directory(searcher, tmp_path):
    (tmp_path / "a.txt").write_text("The sun rises in the east.")
    (tmp_path / "b.txt").write_text("Water boils at 100 degrees.")
    total = searcher.index_path(tmp_path)
    assert total == 2


@requires_embed_model
def test_delete_document(searcher, tmp_path):
    f = tmp_path / "del.txt"
    f.write_text("This will be deleted.")
    searcher.index_path(f)
    deleted = searcher.delete(f)
    assert deleted >= 1
    results = searcher.search("deleted content", top_k=5)
    assert all(r.file_path != str(f) for r in results)


@requires_embed_model
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


@requires_embed_model
def test_search_with_rerank_returns_results(searcher, tmp_path):
    f = tmp_path / "facts.txt"
    f.write_text("Python is a programming language used for data science.")
    searcher.index_path(f)
    # rerank=True but model path won't exist in test env → falls back gracefully
    results = searcher.search("data science language", top_k=1, rerank=True)
    assert len(results) == 1
    assert isinstance(results[0].score, float)
