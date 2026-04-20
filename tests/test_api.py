import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from conftest import requires_embed_model, EMBED_MODEL


@pytest.fixture
def client(tmp_path):
    from knowledge.api.server import create_app
    return TestClient(create_app(chroma_dir=tmp_path / "chroma", embed_model=EMBED_MODEL))


def test_health(tmp_path):
    from knowledge.api.server import create_app
    client = TestClient(create_app(chroma_dir=tmp_path / "chroma", embed_model=EMBED_MODEL))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_nonexistent_path_returns_404(tmp_path):
    from knowledge.api.server import create_app
    client = TestClient(create_app(chroma_dir=tmp_path / "chroma", embed_model=EMBED_MODEL))
    resp = client.post("/index", json={"path": "/nonexistent/path"})
    assert resp.status_code == 404


@requires_embed_model
def test_index_endpoint(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The ocean is vast and deep.")
    resp = client.post("/index", json={"path": str(f)})
    assert resp.status_code == 200
    assert resp.json()["indexed"] >= 1


@requires_embed_model
def test_search_endpoint(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The ocean is vast and deep.")
    client.post("/index", json={"path": str(f)})
    resp = client.post("/search", json={"query": "sea water", "top_k": 1})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert "text" in results[0]
    assert "score" in results[0]
    assert "file_path" in results[0]


@requires_embed_model
def test_search_endpoint_rerank_flag(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The ocean is vast and deep.")
    client.post("/index", json={"path": str(f)})
    resp = client.post("/search", json={"query": "sea water", "top_k": 1, "rerank": True})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@requires_embed_model
def test_delete_endpoint(client, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Temporary content.")
    client.post("/index", json={"path": str(f)})
    resp = client.request("DELETE", "/document", json={"file_path": str(f)})
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 1
