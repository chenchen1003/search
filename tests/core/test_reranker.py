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
