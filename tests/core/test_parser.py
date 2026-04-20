# tests/core/test_parser.py
import textwrap
import pytest
from knowledge.core.parser import parse_file, Chunk


def test_parse_txt_returns_chunks(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Hello world. " * 100)
    chunks = parse_file(f)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.file_path == str(f) for c in chunks)
    assert all(c.file_type == "txt" for c in chunks)


def test_parse_txt_chunk_index_sequential(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("word " * 500)
    chunks = parse_file(f)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_parse_md_splits_by_heading(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Section One\nContent one.\n\n# Section Two\nContent two.\n")
    chunks = parse_file(f)
    assert len(chunks) == 2
    assert "Section One" in chunks[0].text
    assert "Section Two" in chunks[1].text


def test_parse_csv_one_chunk_per_row(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25\n")
    chunks = parse_file(f)
    assert len(chunks) == 2
    assert "name: Alice" in chunks[0].text
    assert "age: 30" in chunks[0].text


def test_parse_json_array_of_objects(tmp_path):
    import json
    f = tmp_path / "items.json"
    f.write_text(json.dumps([{"id": 1, "name": "foo"}, {"id": 2, "name": "bar"}]))
    chunks = parse_file(f)
    assert len(chunks) == 2
    assert "foo" in chunks[0].text


def test_parse_json_single_object(tmp_path):
    import json
    f = tmp_path / "item.json"
    f.write_text(json.dumps({"title": "test", "value": 42}))
    chunks = parse_file(f)
    assert len(chunks) == 1
    assert "title" in chunks[0].text


def test_parse_html_strips_tags(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body><h1>Hello</h1><p>World content here.</p></body></html>"
    )
    chunks = parse_file(f)
    assert len(chunks) > 0
    full_text = " ".join(c.text for c in chunks)
    assert "Hello" in full_text
    assert "World" in full_text
    assert "<h1>" not in full_text


def test_parse_html_file_type_is_html(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>Test.</p></body></html>")
    chunks = parse_file(f)
    assert all(c.file_type == "html" for c in chunks)


def test_parse_unsupported_extension_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(f)


def test_chunk_has_no_catalog_fields(tmp_path):
    """Ensure no domain-specific fields bleed through."""
    f = tmp_path / "notes.txt"
    f.write_text("Some plain text content.")
    chunks = parse_file(f)
    assert len(chunks) > 0
    chunk = chunks[0]
    assert not hasattr(chunk, "audience") or chunk.extra_metadata.get("audience") is None
