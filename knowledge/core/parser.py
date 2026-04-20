from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    extra_metadata: dict = field(default_factory=dict)
    index_text: str | None = None


def parse_file(path: Path) -> list[Chunk]:
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext == "txt":
        return _parse_txt(path)
    if ext == "md":
        return _parse_md(path)
    if ext == "csv":
        return _parse_csv(path)
    if ext == "json":
        return _parse_json(path)
    if ext in ("html", "htm"):
        return _parse_html(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _parse_txt(path: Path, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(
            text=chunk_text,
            file_path=str(path),
            file_type="txt",
            chunk_index=len(chunks),
        ))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _parse_md(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#") and current:
            sections.append("".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current).strip())
    sections = [s for s in sections if s]
    if not sections:
        sections = [text.strip()]
    return [
        Chunk(text=s, file_path=str(path), file_type="md", chunk_index=i)
        for i, s in enumerate(sections)
    ]


def _parse_csv(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = ", ".join(f"{k}: {v}" for k, v in row.items())
            chunks.append(Chunk(text=text, file_path=str(path), file_type="csv", chunk_index=i))
    return chunks


def _parse_json(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    file_path = str(path)
    chunks: list[Chunk] = []

    if isinstance(data, dict) and "documents" in data and "metadata" in data:
        for i, (doc, meta) in enumerate(zip(data["documents"], data["metadata"])):
            text = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
            extra = meta if isinstance(meta, dict) else {}
            chunks.append(Chunk(
                text=text, file_path=file_path, file_type="json",
                chunk_index=i, extra_metadata=extra,
            ))
        return chunks

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = ", ".join(f"{k}: {v}" for k, v in item.items())
                chunks.append(Chunk(
                    text=text, file_path=file_path, file_type="json",
                    chunk_index=i, extra_metadata=item,
                ))
            else:
                chunks.append(Chunk(
                    text=str(item), file_path=file_path,
                    file_type="json", chunk_index=i,
                ))
        return chunks

    if isinstance(data, dict):
        text = ", ".join(f"{k}: {v}" for k, v in data.items())
        chunks.append(Chunk(
            text=text, file_path=file_path, file_type="json",
            chunk_index=0, extra_metadata=data,
        ))
        return chunks

    chunks.append(Chunk(text=str(data), file_path=file_path, file_type="json", chunk_index=0))
    return chunks


def _parse_html(path: Path, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    from langchain_community.document_loaders import BSHTMLLoader
    loader = BSHTMLLoader(str(path), bs_kwargs={"features": "lxml"})
    docs = loader.load()
    text = " ".join(d.page_content for d in docs)
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(
            text=chunk_text,
            file_path=str(path),
            file_type="html",
            chunk_index=len(chunks),
        ))
        if end == len(words):
            break
        start += chunk_size - overlap
    if not chunks:
        chunks.append(Chunk(text=text.strip(), file_path=str(path), file_type="html", chunk_index=0))
    return chunks
