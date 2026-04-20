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
