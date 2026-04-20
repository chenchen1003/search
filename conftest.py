"""
Shared pytest configuration.

Embedding tests skip unless a model is available:
- **Default:** ``models/Qwen3-Embedding-0.6B-Q8_0.gguf`` under the project (GGUF + llama-cpp-python).
- **Override:** ``EMBED_MODEL=...`` (resolved like ``knowledge.config.resolve_embed_model``).
- **HuggingFace:** e.g. ``EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B`` with safetensors in HF cache.
"""

import os
from pathlib import Path

import pytest

from knowledge.config import DEFAULT_EMBED_MODEL, resolve_embed_model

EMBED_MODEL = resolve_embed_model(os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL))

_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-Embedding-0.6B"
    / "snapshots"
)


def _hf_model_ready() -> bool:
    if not _HF_CACHE.exists():
        return False
    snapshots = list(_HF_CACHE.iterdir())
    if not snapshots:
        return False
    return (snapshots[0] / "model.safetensors").exists()


def _model_ready() -> bool:
    p = Path(EMBED_MODEL).expanduser()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return True
    return _hf_model_ready()


def _skip_reason() -> str:
    p = Path(EMBED_MODEL).expanduser()
    if p.suffix.lower() == ".gguf" or EMBED_MODEL.strip().lower().endswith(".gguf"):
        return (
            f"GGUF not found: {EMBED_MODEL}. "
            "Place Qwen3-Embedding-0.6B-Q8_0.gguf in ./models/ or set EMBED_MODEL."
        )
    return (
        f"Model {EMBED_MODEL} not in HuggingFace cache. "
        "Use local GGUF in ./models/ or set EMBED_MODEL."
    )


requires_embed_model = pytest.mark.skipif(
    not _model_ready(),
    reason=_skip_reason(),
)
