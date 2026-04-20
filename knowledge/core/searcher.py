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
