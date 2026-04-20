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
