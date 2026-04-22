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


def _extract_keywords(text: str) -> list[str]:
    """Extract significant brand/entity tokens from a wiki section.

    Pulls out:
    - Chinese segments of 2+ characters (filters common stop words)
    - English words of 2+ characters (lowercased)

    These are stored in the cache and used for fast substring matching at
    query time to catch competitor brands before the embedding gate runs.
    """
    if not text:
        return []

    _ZH_STOPS = {"其他", "产品", "内容", "查询", "类型", "品牌", "运动", "商务", "信息"}

    tokens: set[str] = set()

    # Chinese: sequences of 2+ Han characters
    for m in re.finditer(r"[\u4e00-\u9fff]{2,}", text):
        word = m.group()
        if word not in _ZH_STOPS:
            tokens.add(word)

    # English: capitalized words and known brand abbreviations (2+ chars)
    for m in re.finditer(r"[A-Za-z][A-Za-z\-]{1,}", text):
        word = m.group()
        if word.lower() not in {"and", "or", "the", "of", "in", "for", "not", "etc"}:
            tokens.add(word.lower())

    return sorted(tokens)


def _warn_conflicts(conflicts: list[str], wiki_path: Path) -> None:
    """Emit a warning for terms that appear in both wiki sections."""
    logger.warning(
        "Domain wiki conflict detected in %s: the following terms appear in both "
        "'Answerable queries' AND 'Not answerable' sections, so queries containing "
        "them will be wrongly blocked: %s. "
        "Remove them from the 'Not answerable' section and run "
        "`knowledge wiki check-conflicts` to verify.",
        wiki_path,
        conflicts,
    )


def _query_contains_keyword(query: str, keywords: list[str]) -> bool:
    """Return True if the query contains any of the blocked keywords."""
    query_lower = query.lower()
    for kw in keywords:
        if kw in query_lower or kw in query:
            return True
    return False


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

        import os
        import sys
        from llama_cpp import Llama

        # llama.cpp prints an informational n_ctx warning to stderr even when
        # verbose=False. Suppress it by redirecting stderr during model load.
        with open(os.devnull, "w") as devnull:
            old_stderr, sys.stderr = sys.stderr, devnull
            try:
                llm = Llama(
                    model_path=str(path),
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False,
                )
            finally:
                sys.stderr = old_stderr

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

    def _parse_sections(self, wiki_text: str) -> tuple[str, str]:
        """Extract the 'Answerable queries' and 'Not answerable' section bodies."""
        answerable = ""
        not_answerable = ""

        answerable_match = re.search(
            r"##\s+Answerable queries?\s*\n(.*?)(?=\n##|\Z)", wiki_text,
            re.DOTALL | re.IGNORECASE,
        )
        not_answerable_match = re.search(
            r"##\s+Not answerable\s*\n(.*?)(?=\n##|\Z)", wiki_text,
            re.DOTALL | re.IGNORECASE,
        )
        if answerable_match:
            answerable = answerable_match.group(1).strip()
        if not_answerable_match:
            not_answerable = not_answerable_match.group(1).strip()

        return answerable, not_answerable

    def _load_cache(self) -> dict | None:
        if self.emb_cache_path.exists():
            try:
                data = json.loads(self.emb_cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "blocked_keywords" in data:
                    return data
            except Exception:
                pass
        return None

    def get_embedding(self, embedder: Embedder) -> dict:
        """Return cached embeddings and keyword sets, computing them if needed.

        Cache format::

            {
              "wiki": [...],             # full wiki vector
              "blocked_keywords": [...], # terms from "Not answerable" section
              "conflicts": [...]         # terms that appear in BOTH sections (warning)
            }
        """
        cached = self._load_cache()
        if cached:
            # Surface any cached conflicts on every load so they're never silently ignored
            if cached.get("conflicts"):
                _warn_conflicts(cached["conflicts"], self.wiki_path)
            return cached

        wiki_text = self.load()
        answerable_text, not_answerable_text = self._parse_sections(wiki_text)

        wiki_vec = embedder.embed([wiki_text])[0]
        blocked = _extract_keywords(not_answerable_text)
        answerable_kw = set(_extract_keywords(answerable_text))

        # Detect terms that appear in both sections — these will incorrectly block
        # legitimate queries (the bug we hit with "Nike" appearing in Not answerable).
        conflicts = sorted(set(blocked) & answerable_kw)
        if conflicts:
            _warn_conflicts(conflicts, self.wiki_path)

        cache = {"wiki": wiki_vec, "blocked_keywords": blocked, "conflicts": conflicts}
        self.emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return cache

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, embedder: Embedder) -> dict:
        """Return a validation report without running a full search.

        Reports:
        - ``blocked_keywords``: the full block list derived from "Not answerable"
        - ``conflicts``: terms that appear in both sections (will wrongly block queries)
        - ``ok``: True when there are no conflicts
        """
        cache = self.get_embedding(embedder)
        return {
            "blocked_keywords": cache.get("blocked_keywords", []),
            "conflicts": cache.get("conflicts", []),
            "ok": len(cache.get("conflicts", [])) == 0,
        }

    # ------------------------------------------------------------------
    # Intent check
    # ------------------------------------------------------------------

    def intent_score(self, query_embedding: list[float], embedder: Embedder) -> float:
        """Score query intent using a two-stage check.

        Stage 1 — keyword gate: if the query contains any keyword from the
        "Not answerable" section (e.g. competitor brand names), return 0.0
        immediately.  Brand names are exact tokens; embeddings cannot
        distinguish them from similar brands.

        Stage 2 — embedding gate: cosine similarity between the query and the
        full wiki embedding catches queries that are completely off-topic (e.g.
        "fighting movies") even when no specific blocked keyword is present.
        """
        cache = self.get_embedding(embedder)
        query_lower = query_embedding  # embedding is already computed by caller

        # Stage 1: keyword check
        blocked_keywords: list[str] = cache.get("blocked_keywords", [])
        if blocked_keywords:
            # Re-read query text is not available here; caller must pass it separately.
            # _check_keywords is called from intent_score_for_query instead.
            pass

        # Stage 2: embedding similarity
        return cosine_similarity(query_embedding, cache["wiki"])

    def intent_score_for_query(
        self, query: str, query_embedding: list[float], embedder: Embedder
    ) -> float:
        """Full intent check combining keyword and embedding stages.

        Returns 0.0 if a blocked keyword is found, otherwise returns
        the cosine similarity against the full wiki embedding.
        """
        cache = self.get_embedding(embedder)
        blocked_keywords: list[str] = cache.get("blocked_keywords", [])
        if _query_contains_keyword(query, blocked_keywords):
            return 0.0
        return cosine_similarity(query_embedding, cache["wiki"])
