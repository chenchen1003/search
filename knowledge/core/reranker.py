from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)

_SCORE_PROMPT = """\
You are a search relevance judge. Given a query and a list of passages, score each passage 0-10 for how relevant it is to the query.

Query: {query}

Passages:
{passages}

Reply ONLY with a JSON array, one object per passage: [{{"index": 0, "score": 7}}, ...]
Do not explain. Output JSON only."""


class Reranker:
    def __init__(self, llm_model_path: str) -> None:
        self._model_path = llm_model_path
        self._llm = None

    def _load_llm(self):
        if self._llm is not None:
            return
        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM model not found: {self._model_path}")
        import os
        import sys
        from llama_cpp import Llama

        with open(os.devnull, "w") as devnull:
            old_stderr, sys.stderr = sys.stderr, devnull
            try:
                self._llm = Llama(
                    model_path=str(path),
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False,
                )
            finally:
                sys.stderr = old_stderr

    def _call_llm(self, prompt: str) -> list[dict]:
        self._load_llm()
        response = self._llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
            stop=["```"],
        )
        raw = response["choices"][0]["text"].strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON array in LLM response: {raw!r}")
        return json.loads(match.group())

    def rerank(self, query: str, results: list) -> list:
        if not results:
            return results
        passages = "\n".join(
            f"[{i}] {r.text[:300]}" for i, r in enumerate(results)
        )
        prompt = _SCORE_PROMPT.format(query=query, passages=passages)
        try:
            scores_raw = self._call_llm(prompt)
            score_map = {item["index"]: item["score"] for item in scores_raw}
        except Exception as exc:
            logger.warning("Reranker LLM call failed, using original order: %s", exc)
            return results

        scored = []
        for i, r in enumerate(results):
            llm_score = score_map.get(i, 0)
            scored.append((llm_score / 10.0, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [replace(r, score=round(s, 4)) for s, r in scored]
