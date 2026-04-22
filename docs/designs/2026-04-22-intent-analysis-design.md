# Intent Analysis via Domain Wiki — Design

**Date:** 2026-04-22  
**Status:** Live

---

## Problem

Pure vector search returns results for any query, including completely irrelevant ones. A query like "打斗电影" (fighting movies) against a Nike product catalog returns Nike shoes with moderate confidence scores — the embedding model sees "sportswear" as semantically adjacent to "action". Users expected zero results for out-of-scope queries.

Additionally, vector similarity cannot distinguish brand names. "我要买安踏的鞋子" (I want to buy Anta shoes) and "我要买Nike鞋子" score nearly identically because both describe sports shoes — the brand identity is not captured in the embedding.

---

## Solution

A two-stage intent gate that runs **before** vector search:

**Stage 1 — Keyword gate** (brand / entity names)  
Extract significant terms from the "Not answerable" section of the domain wiki. If any appear in the query, return `[]` immediately. Brand names are exact tokens; embedding similarity cannot distinguish them.

**Stage 2 — Embedding gate** (topic relevance)  
Embed the query and compute cosine similarity against the full wiki embedding. If below `INTENT_THRESHOLD` (default 0.25), return `[]`. This catches completely off-topic queries (e.g. cooking recipes, movies) that contain no specific blocked keyword.

---

## Domain Wiki

A short markdown file (`domain-data/domain.md`) with a fixed three-section structure:

```markdown
# Domain: <title>

## What this index contains
<1-3 sentences describing the indexed data>

## Answerable queries
<bullet list of in-scope query types>

## Not answerable
<bullet list of out-of-scope query types — brand names listed here become the keyword block list>
```

### Generation

`knowledge wiki generate` samples N random chunks from the index, sends them to the local LLM (Qwen2.5-Coder GGUF) with a structured prompt, and saves the output to `domain-data/domain.md`. The file is human-editable; changes take effect immediately after the embedding cache is cleared.

### Embedding cache

`domain-data/domain_emb.json` stores:
- `wiki` — full wiki embedding vector (for the embedding gate)
- `blocked_keywords` — terms extracted from the "Not answerable" section (for the keyword gate)
- `conflicts` — terms that appear in **both** sections (always empty in a valid wiki; populated as a warning)

The cache is invalidated (deleted) when the wiki is saved via the CLI. It is git-ignored; `domain-data/domain.md` is committed.

---

## Conflict Detection

A term in both "Answerable queries" and "Not answerable" will be added to the keyword block list, silently breaking all queries that mention it. This was the root cause of "Nike跑步鞋" being wrongly blocked when "Nike" appeared in a "Not answerable" sentence for context.

**Protection layers:**

1. `DomainWiki.get_embedding()` — detects conflicts at cache-build time and emits `logger.warning` with the conflicting terms.
2. `knowledge wiki validate` — audits the full block list and exits 1 if conflicts exist (CI-friendly).

**Rule:** brand names that belong to the index domain must never appear in the "Not answerable" section text, even in explanatory context like "...与 Nike 无关". Reference them only in the "Answerable queries" section.

---

## Files

| File | Role |
|---|---|
| `knowledge/core/domain_wiki.py` | `DomainWiki` class: generate, load, save, get_embedding, validate, intent_score_for_query |
| `knowledge/core/searcher.py` | `_intent_score()`, `generate_wiki()`, `intent_check` param in `search()` |
| `knowledge/core/index.py` | `sample_chunks(n)` — random chunk sample for wiki generation |
| `knowledge/cli/main.py` | `knowledge wiki generate/show/path/check/validate` sub-commands |
| `knowledge/api/server.py` | `SearchRequest.intent_check` field |
| `knowledge/config.py` | `domain_wiki_path`, `domain_emb_path`, `intent_threshold` |
| `domain-data/domain.md` | Human-readable domain wiki (committed) |
| `domain-data/domain_emb.json` | Embedding + keyword cache (git-ignored) |

---

## Intent Score Reference

| Score | Interpretation |
|---|---|
| `0.0` (keyword match) | Query contains a term from the "Not answerable" block list |
| `< 0.25` | Query is off-topic (embedding gate) |
| `0.25 – 0.45` | Borderline; loosely related to domain |
| `> 0.45` | Clearly in scope |

---

## Operational Workflow

```
After initial index:
  knowledge wiki generate       # one-time setup

After editing domain.md manually:
  knowledge wiki validate       # check for conflicts before searching
  knowledge wiki check "query"  # spot-check specific queries

After a major data change:
  knowledge index ./new-data --update-wiki   # index + regenerate in one step
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Wiki lives in `domain-data/`, not `chroma/` | Separates human-curated content from auto-generated index; makes it committable without committing the vector store |
| Default threshold 0.25 | Empirically validated: Nike domain queries score 0.35–0.69, off-topic queries score 0.10–0.24 |
| Wiki generation is explicit, not auto | LLM call takes ~25s; users should control when it runs; auto-generation on every index would be too slow |
| `Searcher` defaults wiki path from `settings`, not `chroma_dir` | After the path was moved, hardcoding `chroma_dir / "domain.md"` caused the intent gate to be silently skipped on every search |
| Fallback: no wiki → intent check skipped | Backward-compatible; existing setups work without generating a wiki |
