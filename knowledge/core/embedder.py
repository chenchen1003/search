from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
from typing import Any, Iterator

from knowledge.config import resolve_embed_model


def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0.0:
        return vec
    return [x / s for x in vec]


def is_gguf_model(model_name: str) -> bool:
    p = Path(model_name).expanduser()
    if model_name.lower().endswith(".gguf"):
        return True
    return p.is_file() and p.suffix.lower() == ".gguf"


@contextlib.contextmanager
def _suppress_llama_cpp_stderr() -> Iterator[None]:
    """Hide llama.cpp ``init: embeddings required…`` spam on stderr (C++ logs ignore ``verbose=False``).

    Set ``KNOWLEDGE_LLAMA_VERBOSE=1`` to see full stderr again.
    """
    if os.environ.get("KNOWLEDGE_LLAMA_VERBOSE", "").lower() in ("1", "true", "yes"):
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


class Embedder:
    """Text → vectors via SentenceTransformer (HF id or local ST folder) or GGUF (local ``.gguf`` file)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = resolve_embed_model(model_name)
        self._st_model: Any = None
        self._llama: Any = None
        self._max_chars = 0

    def _load(self) -> None:
        if self._st_model is not None or self._llama is not None:
            return

        if is_gguf_model(self._model_name):
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python is required for .gguf embedding models. "
                    "Install: pip install llama-cpp-python\n"
                    "Apple Silicon (faster): "
                    'CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir'
                ) from e

            path = str(Path(self._model_name).expanduser().resolve())
            if not Path(path).is_file():
                raise FileNotFoundError(f"GGUF model file not found: {path}")

            n_ctx = int(os.environ.get("KNOWLEDGE_LLAMA_N_CTX", "8192"))
            n_gpu = int(os.environ.get("KNOWLEDGE_LLAMA_N_GPU_LAYERS", "-1"))
            verbose = os.environ.get("KNOWLEDGE_LLAMA_VERBOSE", "").lower() in ("1", "true", "yes")

            with _suppress_llama_cpp_stderr():
                self._llama = Llama(
                    model_path=path,
                    embedding=True,
                    verbose=verbose,
                    n_ctx=n_ctx,
                    n_batch=min(512, n_ctx),
                    n_gpu_layers=n_gpu,
                )
            self._max_chars = max(512, n_ctx * 3)
        else:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()

        if self._llama is not None:
            out: list[list[float]] = []
            with _suppress_llama_cpp_stderr():
                for t in texts:
                    chunk = t if len(t) <= self._max_chars else t[: self._max_chars]
                    r = self._llama.create_embedding(chunk)
                    vec = [float(x) for x in r["data"][0]["embedding"]]
                    out.append(_l2_normalize(vec))
            return out

        assert self._st_model is not None
        vectors = self._st_model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
