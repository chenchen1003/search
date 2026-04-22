from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from knowledge.config import settings
from knowledge.core.searcher import Searcher

app = typer.Typer(help="Local knowledge semantic search engine.")
wiki_app = typer.Typer(help="Manage the domain knowledge wiki used for intent analysis.")
app.add_typer(wiki_app, name="wiki")


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
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Combine vector + BM25 keyword scoring (default: on)"),
    intent: bool = typer.Option(True, "--intent/--no-intent", help="Check query intent against domain wiki before searching (default: on)"),
    min_score: float = typer.Option(0.45, "--min-score", help="Minimum raw vector similarity (0–1); results below this are suppressed"),
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
) -> None:
    """Semantic search over indexed files."""
    searcher = _make_searcher(chroma_dir, model)
    results = searcher.search(
        query,
        top_k=top_k,
        file_type=file_type,
        rerank=rerank,
        hybrid=hybrid,
        min_score=min_score,
        intent_check=intent,
    )
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


# ---------------------------------------------------------------------------
# knowledge wiki sub-commands
# ---------------------------------------------------------------------------

@wiki_app.command("generate")
def wiki_generate(
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
    llm_model: str = typer.Option(settings.llm_model, help="LLM model path for wiki generation"),
    n_samples: int = typer.Option(10, "--samples", help="Number of random chunks to sample"),
    wiki_path: Path = typer.Option(settings.domain_wiki_path, help="Path to save domain wiki"),
) -> None:
    """Generate the domain wiki by sampling indexed chunks and asking the local LLM."""
    typer.echo(f"Sampling {n_samples} chunks from index...")
    searcher = Searcher(
        chroma_dir=chroma_dir, embed_model=model, llm_model=llm_model,
        wiki_path=wiki_path, emb_cache_path=wiki_path.parent / "domain_emb.json",
    )
    try:
        content = searcher.generate_wiki(n_samples=n_samples)
    except (RuntimeError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"\nWiki saved to {wiki_path}\n")
    typer.echo(content)


@wiki_app.command("show")
def wiki_show(
    wiki_path: Path = typer.Option(settings.domain_wiki_path, help="Path to domain wiki"),
) -> None:
    """Print the current domain wiki."""
    if not wiki_path.exists():
        typer.echo(f"No domain wiki found at {wiki_path}. Run: knowledge wiki generate", err=True)
        raise typer.Exit(1)
    typer.echo(wiki_path.read_text(encoding="utf-8"))


@wiki_app.command("path")
def wiki_path_cmd(
    wiki_path: Path = typer.Option(settings.domain_wiki_path, help="Path to domain wiki"),
) -> None:
    """Print the path to the domain wiki file (useful for shell scripting)."""
    typer.echo(wiki_path)


if __name__ == "__main__":
    app()
