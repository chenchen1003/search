from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = directory that contains the ``knowledge/`` package (the ``search`` worktree).
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Default: GGUF next to this repo (see README). Resolved against ``PACKAGE_ROOT`` so CLI works from any cwd.
DEFAULT_EMBED_MODEL = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"


def resolve_embed_model(model: str) -> str:
    """Turn env/CLI model strings into absolute paths when they live under ``PACKAGE_ROOT``.

    - Absolute paths: expanded and returned as-is.
    - Relative ``*.gguf`` or any relative path that **exists** under ``PACKAGE_ROOT``: anchored there.
    - Missing ``*.gguf`` under project: still anchored to ``PACKAGE_ROOT`` (clear ``FileNotFoundError`` later).
    - Values with ``/`` that are not ``.gguf`` and do not exist as paths: treated as HuggingFace repo ids (e.g. ``Qwen/Qwen3-Embedding-0.6B``).
    """
    raw = (model or "").strip()
    if not raw:
        return raw
    p = Path(raw)

    if p.is_absolute():
        return str(p.expanduser().resolve())

    rel = raw.removeprefix("./")
    lowered = rel.lower()
    candidate = (PACKAGE_ROOT / rel).resolve()

    if candidate.exists():
        return str(candidate)

    if lowered.endswith(".gguf"):
        return str(candidate)

    if "/" in rel:
        return raw

    return raw


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    chroma_dir: Path = PACKAGE_ROOT / "chroma"
    embed_model: str = DEFAULT_EMBED_MODEL
    llm_model: str = str(PACKAGE_ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf")
    chunk_size: int = 400
    chunk_overlap: int = 50
    api_port: int = 8000

    @model_validator(mode="after")
    def _resolve_embed_model(self) -> "Settings":
        self.embed_model = resolve_embed_model(self.embed_model)
        return self


settings = Settings()
