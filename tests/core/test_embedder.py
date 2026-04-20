import pytest
from conftest import requires_embed_model, EMBED_MODEL
from knowledge.config import PACKAGE_ROOT, resolve_embed_model
from knowledge.core.embedder import Embedder, is_gguf_model


def test_resolve_embed_model_anchors_relative_gguf():
    resolved = resolve_embed_model("models/Qwen3-Embedding-0.6B-Q8_0.gguf")
    assert resolved.startswith(str(PACKAGE_ROOT))
    assert resolved.endswith("Qwen3-Embedding-0.6B-Q8_0.gguf")


def test_resolve_embed_model_hf_id_unchanged():
    assert resolve_embed_model("Qwen/Qwen3-Embedding-0.6B") == "Qwen/Qwen3-Embedding-0.6B"


def test_is_gguf_model_detects_suffix():
    assert is_gguf_model("/tmp/foo.gguf") is True
    assert is_gguf_model("Qwen/Qwen3-Embedding-0.6B") is False


def test_is_gguf_model_detects_existing_file(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"dummy")
    assert is_gguf_model(str(f)) is True


@requires_embed_model
def test_embed_returns_list_of_floats():
    embedder = Embedder(model_name=EMBED_MODEL)
    vectors = embedder.embed(["hello world"])
    assert len(vectors) == 1
    assert isinstance(vectors[0], list)
    assert all(isinstance(v, float) for v in vectors[0])


@requires_embed_model
def test_embed_batch_returns_correct_count():
    embedder = Embedder(model_name=EMBED_MODEL)
    texts = ["first", "second", "third"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3


@requires_embed_model
def test_embed_same_text_produces_same_vector():
    embedder = Embedder(model_name=EMBED_MODEL)
    v1 = embedder.embed(["test sentence"])[0]
    v2 = embedder.embed(["test sentence"])[0]
    assert v1 == v2


@requires_embed_model
def test_embed_different_texts_produce_different_vectors():
    embedder = Embedder(model_name=EMBED_MODEL)
    v1 = embedder.embed(["cat"])[0]
    v2 = embedder.embed(["quantum physics"])[0]
    assert v1 != v2
