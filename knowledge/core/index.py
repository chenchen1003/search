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

    def sample_chunks(self, n: int = 30) -> list[str]:
        """Return up to n random document texts from the collection for domain profiling."""
        import random
        total = self._collection.count()
        if total == 0:
            return []
        all_ids = self._collection.get(include=[])["ids"]
        sample_ids = random.sample(all_ids, min(n, len(all_ids)))
        result = self._collection.get(ids=sample_ids, include=["documents"])
        return result["documents"] or []

    @property
    def embedder(self) -> "Embedder":
        return self._embedder
