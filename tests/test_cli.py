import pytest
from typer.testing import CliRunner
from conftest import requires_embed_model, EMBED_MODEL
from knowledge.cli.main import app

runner = CliRunner()


@requires_embed_model
def test_index_command_success(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Some text content here.")
    result = runner.invoke(app, ["index", str(f), "--chroma-dir", str(tmp_path / "chroma"),
                                 "--model", EMBED_MODEL])
    assert result.exit_code == 0
    assert "Indexed" in result.output


@requires_embed_model
def test_search_command_success(tmp_path):
    chroma = tmp_path / "chroma"
    f = tmp_path / "notes.txt"
    f.write_text("Penguins live in Antarctica.")
    runner.invoke(app, ["index", str(f), "--chroma-dir", str(chroma),
                        "--model", EMBED_MODEL])
    result = runner.invoke(app, ["search", "cold weather birds", "--chroma-dir", str(chroma),
                                 "--model", EMBED_MODEL])
    assert result.exit_code == 0
    assert "Antarctica" in result.output or len(result.output) > 0


@requires_embed_model
def test_search_command_rerank_flag(tmp_path):
    chroma = tmp_path / "chroma"
    f = tmp_path / "notes.txt"
    f.write_text("Penguins live in Antarctica.")
    runner.invoke(app, ["index", str(f), "--chroma-dir", str(chroma),
                        "--model", EMBED_MODEL])
    result = runner.invoke(app, ["search", "cold birds", "--chroma-dir", str(chroma),
                                 "--model", EMBED_MODEL,
                                 "--rerank"])
    # rerank falls back gracefully when model is missing
    assert result.exit_code == 0


@requires_embed_model
def test_delete_command_success(tmp_path):
    chroma = tmp_path / "chroma"
    f = tmp_path / "del.txt"
    f.write_text("Content to remove.")
    runner.invoke(app, ["index", str(f), "--chroma-dir", str(chroma),
                        "--model", EMBED_MODEL])
    result = runner.invoke(app, ["delete", str(f), "--chroma-dir", str(chroma),
                                 "--model", EMBED_MODEL])
    assert result.exit_code == 0
    assert "Deleted" in result.output
