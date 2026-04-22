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
    update_wiki: bool = typer.Option(False, "--update-wiki", help="Regenerate domain wiki after indexing (requires LLM)"),
) -> None:
    """Index a file or directory."""
    searcher = _make_searcher(chroma_dir, model)
    total = searcher.index_path(path)
    typer.echo(f"Indexed {total} chunk(s) from {path}")

    if update_wiki:
        typer.echo("Regenerating domain wiki...")
        try:
            content = searcher.generate_wiki()
            typer.echo(f"Wiki updated at {settings.domain_wiki_path}")
        except (RuntimeError, FileNotFoundError) as e:
            typer.echo(f"Warning: wiki generation failed — {e}", err=True)
    elif settings.domain_wiki_path.exists():
        typer.echo(
            f"Tip: domain wiki may be outdated. Run `knowledge wiki generate` to refresh it."
        )


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


@wiki_app.command("validate")
def wiki_validate(
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
    wiki_path: Path = typer.Option(settings.domain_wiki_path, help="Path to domain wiki"),
) -> None:
    """Validate the domain wiki: show the block list and detect section conflicts.

    Conflicts occur when a term appears in both 'Answerable queries' and 'Not
    answerable' — those terms will incorrectly block legitimate queries.
    Remove them from the 'Not answerable' section to fix.
    """
    if not wiki_path.exists():
        typer.echo(f"No domain wiki found at {wiki_path}. Run: knowledge wiki generate", err=True)
        raise typer.Exit(1)

    from knowledge.core.domain_wiki import DomainWiki
    import json as _json

    # Delete cache to force a fresh analysis from the current wiki file
    emb_cache = wiki_path.parent / "domain_emb.json"
    if emb_cache.exists():
        emb_cache.unlink()

    searcher = Searcher(chroma_dir=chroma_dir, embed_model=model,
                        wiki_path=wiki_path, emb_cache_path=emb_cache)
    wiki = DomainWiki(wiki_path, emb_cache)
    report = wiki.validate(searcher._index.embedder)

    typer.echo(f"\nBlocked keywords ({len(report['blocked_keywords'])}):")
    for kw in report["blocked_keywords"]:
        typer.echo(f"  {kw}")

    if report["conflicts"]:
        typer.echo(
            "\n" + typer.style("CONFLICTS DETECTED", fg=typer.colors.RED, bold=True)
            + f" ({len(report['conflicts'])} term(s) appear in BOTH sections"
            " and will wrongly block legitimate queries):"
        )
        for kw in report["conflicts"]:
            typer.echo(f"  {typer.style(kw, fg=typer.colors.RED)}  ← remove from 'Not answerable'")
        raise typer.Exit(1)
    else:
        typer.echo("\n" + typer.style("OK", fg=typer.colors.GREEN, bold=True)
                   + " — no conflicts detected.")


@wiki_app.command("check")
def wiki_check(
    query: str = typer.Argument(..., help="Query to test against the domain wiki"),
    chroma_dir: Path = typer.Option(settings.chroma_dir, help="ChromaDB storage path"),
    model: str = typer.Option(settings.embed_model, help="Embedding model name"),
    wiki_path: Path = typer.Option(settings.domain_wiki_path, help="Path to domain wiki"),
    threshold: float = typer.Option(settings.intent_threshold, "--threshold", help="Intent threshold (default from config)"),
) -> None:
    """Check whether a query matches the domain wiki (intent gate dry-run)."""
    if not wiki_path.exists():
        typer.echo(f"No domain wiki found at {wiki_path}. Run: knowledge wiki generate", err=True)
        raise typer.Exit(1)

    searcher = Searcher(
        chroma_dir=chroma_dir, embed_model=model,
        wiki_path=wiki_path,
        emb_cache_path=wiki_path.parent / "domain_emb.json",
        intent_threshold=threshold,
    )
    score = searcher._intent_score(query)
    passed = score >= threshold

    status = typer.style("PASS", fg=typer.colors.GREEN, bold=True) if passed else typer.style("BLOCK", fg=typer.colors.RED, bold=True)
    typer.echo(f"{status}  score={score:.3f}  threshold={threshold}  query={query!r}")
    if not passed:
        typer.echo(f"  → Query is outside the domain. Search would return no results.")
    else:
        typer.echo(f"  → Query is within the domain. Search will proceed normally.")


if __name__ == "__main__":
    app()
