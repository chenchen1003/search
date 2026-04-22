from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from knowledge.core.index import VectorIndex
from knowledge.core.parser import parse_file
from knowledge.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    score: float
    extra_metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".html", ".htm"}

# Reciprocal Rank Fusion constant — higher k reduces the impact of top ranks
_RRF_K = 60
# How many extra candidates to fetch for hybrid/rerank passes
_HYBRID_FETCH_MULTIPLIER = 4


def _tokenize_zh(text: str) -> list[str]:
    """Tokenize Chinese (and mixed) text using jieba word segmentation."""
    import jieba
    # Silence jieba's "Building prefix dict..." startup logs
    jieba.setLogLevel(logging.ERROR)
    return [t for t in jieba.cut(text) if t.strip()]


def _rrf_combine(
    vector_results: list[SearchResult],
    bm25_scores: list[float],
    top_k: int,
) -> list[SearchResult]:
    """
    Merge vector ranks and BM25 scores using Reciprocal Rank Fusion.
    vector_results and bm25_scores must share the same index ordering.
    """
    # Build BM25 rank by sorting indices by descending bm25 score
    bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order)}

    combined: list[tuple[float, SearchResult]] = []
    for vec_rank, result in enumerate(vector_results):
        rrf = 1.0 / (_RRF_K + vec_rank + 1) + 1.0 / (_RRF_K + bm25_rank[vec_rank] + 1)
        combined.append((rrf, result))

    combined.sort(key=lambda x: x[0], reverse=True)
    # Normalise RRF score to [0, 1] so it stays comparable with old scalar scores
    max_rrf = combined[0][0] if combined else 1.0
    return [replace(r, score=round(s / max_rrf, 4)) for s, r in combined[:top_k]]


class Searcher:
    def __init__(
        self,
        chroma_dir: Path,
        embed_model: str,
        llm_model: str | None = None,
        wiki_path: Path | None = None,
        emb_cache_path: Path | None = None,
        intent_threshold: float | None = None,
    ) -> None:
        self._index = VectorIndex(chroma_dir=chroma_dir, embed_model=embed_model)
        self._llm_model = llm_model or settings.llm_model
        chroma_dir = Path(chroma_dir)
        self._wiki_path = wiki_path or (chroma_dir / "domain.md")
        self._emb_cache_path = emb_cache_path or (chroma_dir / "domain_emb.json")
        self._intent_threshold = intent_threshold if intent_threshold is not None else settings.intent_threshold

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

    def generate_wiki(self, n_samples: int = 10) -> str:
        """Sample chunks from the index, call the LLM, persist wiki + clear emb cache."""
        from knowledge.core.domain_wiki import DomainWiki
        wiki = DomainWiki(self._wiki_path, self._emb_cache_path)
        chunks = self._index.sample_chunks(n_samples)
        if not chunks:
            raise RuntimeError("Index is empty — run 'knowledge index' first.")
        content = wiki.generate(chunks, self._llm_model)
        wiki.save(content)
        return content

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_type: str | None = None,
        rerank: bool = False,
        hybrid: bool = True,
        min_score: float = 0.45,
        intent_check: bool = True,
    ) -> list[SearchResult]:
        # Intent gate: compare query embedding against domain wiki embedding.
        # Runs only when domain.md exists; fails silently so existing setups
        # are unaffected until the user runs `knowledge wiki generate`.
        if intent_check and self._wiki_path.exists():
            score = self._intent_score(query)
            if score < self._intent_threshold:
                logger.debug(
                    "Intent check rejected query %r (score %.3f < threshold %.3f)",
                    query, score, self._intent_threshold,
                )
                return []

        # Fetch a larger candidate pool when hybrid or rerank passes are needed
        fetch_k = top_k * _HYBRID_FETCH_MULTIPLIER if (hybrid or rerank) else top_k
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

        # Filter by raw cosine similarity BEFORE RRF normalization can hide low relevance.
        # Good domain matches score ~0.5–0.7; spurious cross-domain matches score ~0.3–0.4.
        # Filtering individually removes weak candidates and returns nothing when the whole
        # query is off-topic (no result clears the bar).
        if min_score > 0 and results:
            results = [r for r in results if r.score >= min_score]
            if not results:
                return []

        if hybrid and results:
            results = self._hybrid_rerank(query, results, top_k)

        if rerank and results:
            from knowledge.core.reranker import Reranker
            reranker = Reranker(llm_model_path=self._llm_model)
            results = reranker.rerank(query, results)

        return results[:top_k]

    def _intent_score(self, query: str) -> float:
        """Embed the query and run the two-stage intent check (keyword + embedding)."""
        from knowledge.core.domain_wiki import DomainWiki
        wiki = DomainWiki(self._wiki_path, self._emb_cache_path)
        query_vec = self._index.embedder.embed([query])[0]
        return wiki.intent_score_for_query(query, query_vec, self._index.embedder)

    def _hybrid_rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Re-rank candidates by combining vector rank with BM25 keyword score (RRF)."""
        from rank_bm25 import BM25Okapi

        query_tokens = _tokenize_zh(query)
        corpus = [_tokenize_zh(r.text) for r in results]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens).tolist()
        return _rrf_combine(results, bm25_scores, top_k)

    def delete(self, path: Path) -> int:
        return self._index.delete(str(path))
