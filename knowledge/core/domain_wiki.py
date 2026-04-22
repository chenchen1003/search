from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.core.embedder import Embedder

logger = logging.getLogger(__name__)

_WIKI_PROMPT = """\
You are a knowledge base analyst. Below are sample records from a search index.
Write a concise domain wiki (80-150 words) in the SAME LANGUAGE as the sample records.
Use this exact format:

# Domain: <short title>

## What this index contains
<1-3 sentences describing the data>

## Answerable queries
<bullet list: 4-6 concrete query types that will find results>

## Not answerable
<bullet list: 3-5 out-of-scope query types>

Sample records:
{chunks}

Reply with the wiki ONLY. No explanation. Stop after the "Not answerable" section."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-math cosine similarity — no external deps required."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class DomainWiki:
    """Manages the LLM-generated domain knowledge wiki for intent gating."""

    def __init__(self, wiki_path: Path, emb_cache_path: Path) -> None:
        self.wiki_path = Path(wiki_path)
        self.emb_cache_path = Path(emb_cache_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self.wiki_path.exists()

    def load(self) -> str:
        return self.wiki_path.read_text(encoding="utf-8")

    def save(self, content: str) -> None:
        self.wiki_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_path.write_text(content, encoding="utf-8")
        # Invalidate cached embedding whenever the wiki changes
        if self.emb_cache_path.exists():
            self.emb_cache_path.unlink()

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def generate(self, chunk_texts: list[str], llm_model_path: str) -> str:
        """Call the local LLM to produce a domain wiki from sampled chunk texts."""
        path = Path(llm_model_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM model not found: {llm_model_path}")

        from llama_cpp import Llama

        llm = Llama(
            model_path=str(path),
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )

        # Use a tight sample — 200 chars per chunk keeps the prompt well within 4096 tokens
        sample_text = "\n---\n".join(t[:200] for t in chunk_texts)
        prompt = _WIKI_PROMPT.format(chunks=sample_text)

        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["```", "\n\n\n"],
        )
        raw = response["choices"][0]["text"].strip()

        # Extract the wiki starting from the first markdown heading
        match = re.search(r"(# Domain:.*)", raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: wrap the raw output in a minimal wiki structure so the file
        # is still useful as a semantic anchor for intent gating.
        return f"# Domain: (auto-generated)\n\n## What this index contains\n{raw[:600]}"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def get_embedding(self, embedder: Embedder) -> list[float]:
        """Return the cached wiki embedding, computing it if needed."""
        if self.emb_cache_path.exists():
            try:
                return json.loads(self.emb_cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass  # Cache corrupt — recompute

        wiki_text = self.load()
        vec = embedder.embed([wiki_text])[0]
        self.emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_cache_path.write_text(json.dumps(vec), encoding="utf-8")
        return vec

    # ------------------------------------------------------------------
    # Intent check
    # ------------------------------------------------------------------

    def intent_score(self, query_embedding: list[float], embedder: Embedder) -> float:
        """Return cosine similarity between the query and the domain wiki."""
        wiki_emb = self.get_embedding(embedder)
        return cosine_similarity(query_embedding, wiki_emb)
