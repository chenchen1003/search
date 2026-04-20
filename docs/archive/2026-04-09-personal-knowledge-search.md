# Personal Knowledge Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a personal knowledge base with RAG chat — ingest mixed content (markdown, PDFs, web pages, code, images, CSV/JSON) and answer questions via a web UI with source citations.

**Architecture:** Python FastAPI backend with LlamaIndex for document loading/RAG and ChromaDB for vector storage. React frontend with TailwindCSS for a chat UI with streaming answers and file upload. OpenAI API for embeddings and chat completions.

**Tech Stack:** Python 3.11+, FastAPI, LlamaIndex, ChromaDB, OpenAI API, React 18, Vite, TailwindCSS, SQLite

---

### Task 1: Backend Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`

**Step 1: Create `backend/requirements.txt`**

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-dotenv==1.0.1
python-multipart==0.0.18
llama-index==0.12.5
llama-index-vector-stores-chroma==0.4.1
llama-index-readers-file==0.4.3
llama-index-embeddings-openai==0.3.1
llama-index-llms-openai==0.3.18
chromadb==0.6.3
openai==1.61.0
beautifulsoup4==4.12.3
pydantic==2.10.4
aiosqlite==0.20.0
aiofiles==24.1.0
```

Note: Check latest versions with `pip index versions <package>` if install fails. Use compatible versions.

**Step 2: Create `backend/.env.example`**

```
OPENAI_API_KEY=sk-your-key-here
DATA_DIR=./data
CHROMA_DIR=./data/chroma
UPLOAD_DIR=./data/uploads
CHAT_DB=./data/chat.db
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

**Step 3: Create `backend/app/config.py`**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    upload_dir: Path = Path("./data/uploads")
    chat_db: Path = Path("./data/chat.db")
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
```

**Step 4: Create `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

app = FastAPI(title="Local Knowledge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

**Step 5: Create empty `backend/app/__init__.py`**

Empty file.

**Step 6: Install dependencies and verify server starts**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` with a real `OPENAI_API_KEY`.

```bash
cd backend
cp .env.example .env
# Edit .env with your actual OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

Expected: Server starts, `GET http://localhost:8000/api/health` returns `{"status": "ok"}`

**Step 7: Commit**

```bash
git add backend/
git commit -m "feat: scaffold backend with FastAPI, config, and health check"
```

---

### Task 2: Storage Service (ChromaDB + SQLite)

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/storage.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/schemas.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_storage.py`

**Step 1: Create `backend/app/models/schemas.py`**

```python
from pydantic import BaseModel
from datetime import datetime


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    file_type: str
    source_path: str
    chunk_count: int
    ingested_at: datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    sources: list[dict] | None = None


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: datetime
    messages: list[ChatMessage] = []


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class SourceCitation(BaseModel):
    document_name: str
    chunk_text: str
    score: float
```

**Step 2: Create `backend/app/services/storage.py`**

```python
import uuid
import json
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.models.schemas import DocumentMetadata, ChatSession, ChatMessage


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, doc_id: str, chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def delete_document(self, doc_id: str):
        existing = self.collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def count(self) -> int:
        return self.collection.count()


class ChatStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or settings.chat_db)

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            await db.commit()

    async def create_session(self, title: str = "New Chat") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                (session_id, title, now),
            )
            await db.commit()
        return session_id

    async def add_message(self, session_id: str, role: str, content: str, sources: list[dict] | None = None):
        now = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources) if sources else None
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, sources_json, now),
            )
            await db.commit()

    async def get_session(self, session_id: str) -> ChatSession | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
            if not row:
                return None

            msg_cursor = await db.execute(
                "SELECT role, content, sources FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
            messages = []
            async for msg in msg_cursor:
                sources = json.loads(msg["sources"]) if msg["sources"] else None
                messages.append(ChatMessage(role=msg["role"], content=msg["content"], sources=sources))

            return ChatSession(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]),
                messages=messages,
            )

    async def list_sessions(self) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_session_title(self, session_id: str, title: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            await db.commit()


class DocumentStore:
    """Tracks ingested document metadata in SQLite."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or settings.chat_db)

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                )
            """)
            await db.commit()

    async def add_document(self, doc: DocumentMetadata):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO documents (id, filename, file_type, source_path, chunk_count, ingested_at) VALUES (?, ?, ?, ?, ?, ?)",
                (doc.id, doc.filename, doc.file_type, doc.source_path, doc.chunk_count, doc.ingested_at.isoformat()),
            )
            await db.commit()

    async def list_documents(self) -> list[DocumentMetadata]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
            rows = await cursor.fetchall()
            return [
                DocumentMetadata(
                    id=row["id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    source_path=row["source_path"],
                    chunk_count=row["chunk_count"],
                    ingested_at=datetime.fromisoformat(row["ingested_at"]),
                )
                for row in rows
            ]

    async def delete_document(self, doc_id: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def get_document(self, doc_id: str) -> DocumentMetadata | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = await cursor.fetchone()
            if not row:
                return None
            return DocumentMetadata(
                id=row["id"],
                filename=row["filename"],
                file_type=row["file_type"],
                source_path=row["source_path"],
                chunk_count=row["chunk_count"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
            )
```

**Step 3: Write tests for storage**

Create `backend/tests/test_storage.py`:

```python
import pytest
import tempfile
from pathlib import Path

from app.services.storage import VectorStore, ChatStore, DocumentStore
from app.models.schemas import DocumentMetadata
from datetime import datetime, timezone


@pytest.fixture
def chat_store(tmp_path):
    store = ChatStore(db_path=tmp_path / "test_chat.db")
    return store


@pytest.fixture
def doc_store(tmp_path):
    store = DocumentStore(db_path=tmp_path / "test_docs.db")
    return store


@pytest.mark.asyncio
async def test_chat_store_create_and_get_session(chat_store):
    await chat_store.init_db()
    session_id = await chat_store.create_session("Test Chat")
    assert session_id is not None

    await chat_store.add_message(session_id, "user", "Hello")
    await chat_store.add_message(session_id, "assistant", "Hi there!", [{"doc": "test.md", "text": "snippet"}])

    session = await chat_store.get_session(session_id)
    assert session is not None
    assert session.title == "Test Chat"
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].sources is not None


@pytest.mark.asyncio
async def test_chat_store_list_sessions(chat_store):
    await chat_store.init_db()
    await chat_store.create_session("Chat 1")
    await chat_store.create_session("Chat 2")
    sessions = await chat_store.list_sessions()
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_document_store_crud(doc_store):
    await doc_store.init_db()
    doc = DocumentMetadata(
        id="test-123",
        filename="test.pdf",
        file_type="pdf",
        source_path="/uploads/test.pdf",
        chunk_count=5,
        ingested_at=datetime.now(timezone.utc),
    )
    await doc_store.add_document(doc)

    docs = await doc_store.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "test.pdf"

    fetched = await doc_store.get_document("test-123")
    assert fetched is not None
    assert fetched.chunk_count == 5

    deleted = await doc_store.delete_document("test-123")
    assert deleted is True

    docs = await doc_store.list_documents()
    assert len(docs) == 0
```

**Step 4: Run tests**

```bash
cd backend
pip install pytest pytest-asyncio
python -m pytest tests/test_storage.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add backend/app/services/ backend/app/models/ backend/tests/
git commit -m "feat: add storage services (ChromaDB, SQLite chat/document stores)"
```

---

### Task 3: Ingestion Service

**Files:**
- Create: `backend/app/services/ingestion.py`
- Create: `backend/tests/test_ingestion.py`

**Step 1: Create `backend/app/services/ingestion.py`**

```python
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timezone

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import settings
from app.models.schemas import DocumentMetadata, IngestResponse
from app.services.storage import VectorStore, DocumentStore

SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".pdf", ".html", ".htm",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".csv", ".json",
    ".png", ".jpg", ".jpeg",
}


def get_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    type_map = {
        ".md": "markdown", ".txt": "text",
        ".pdf": "pdf",
        ".html": "web", ".htm": "web",
        ".csv": "csv", ".json": "json",
        ".png": "image", ".jpg": "image", ".jpeg": "image",
    }
    if ext in type_map:
        return type_map[ext]
    if ext in {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h"}:
        return "code"
    return "unknown"


class IngestionService:
    def __init__(self, vector_store: VectorStore, doc_store: DocumentStore):
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.embed_model = OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def ingest_file(self, file_path: Path, original_filename: str | None = None) -> IngestResponse:
        filename = original_filename or file_path.name
        file_type = get_file_type(file_path)

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        documents = reader.load_data()

        if not documents:
            raise ValueError(f"No content extracted from {filename}")

        nodes = self.splitter.get_nodes_from_documents(documents)
        chunks = [node.get_content() for node in nodes]

        embeddings = []
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = await self.embed_model.aget_text_embedding_batch(batch)
            embeddings.extend(batch_embeddings)

        doc_id = str(uuid.uuid4())
        metadatas = [
            {"doc_id": doc_id, "filename": filename, "file_type": file_type, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.vector_store.add_chunks(doc_id, chunks, embeddings, metadatas)

        doc_meta = DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_type=file_type,
            source_path=str(file_path),
            chunk_count=len(chunks),
            ingested_at=datetime.now(timezone.utc),
        )
        await self.doc_store.add_document(doc_meta)

        return IngestResponse(
            document_id=doc_id,
            filename=filename,
            chunk_count=len(chunks),
            status="success",
        )

    async def ingest_uploaded_file(self, file_content: bytes, filename: str) -> IngestResponse:
        dest = settings.upload_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            dest = settings.upload_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

        dest.write_bytes(file_content)
        return await self.ingest_file(dest, filename)

    async def ingest_directory(self, directory: Path) -> list[IngestResponse]:
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        results = []
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    result = await self.ingest_file(file_path)
                    results.append(result)
                except Exception as e:
                    results.append(IngestResponse(
                        document_id="",
                        filename=file_path.name,
                        chunk_count=0,
                        status=f"error: {str(e)}",
                    ))
        return results

    async def ingest_url(self, url: str) -> IngestResponse:
        from llama_index.readers.web import BeautifulSoupWebReader

        reader = BeautifulSoupWebReader()
        documents = reader.load_data(urls=[url])

        if not documents:
            raise ValueError(f"No content extracted from {url}")

        nodes = self.splitter.get_nodes_from_documents(documents)
        chunks = [node.get_content() for node in nodes]

        embeddings = await self.embed_model.aget_text_embedding_batch(chunks)

        doc_id = str(uuid.uuid4())
        metadatas = [
            {"doc_id": doc_id, "filename": url, "file_type": "web", "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.vector_store.add_chunks(doc_id, chunks, embeddings, metadatas)

        doc_meta = DocumentMetadata(
            id=doc_id,
            filename=url,
            file_type="web",
            source_path=url,
            chunk_count=len(chunks),
            ingested_at=datetime.now(timezone.utc),
        )
        await self.doc_store.add_document(doc_meta)

        return IngestResponse(
            document_id=doc_id,
            filename=url,
            chunk_count=len(chunks),
            status="success",
        )
```

**Step 2: Write test for ingestion (using a simple text file, no OpenAI needed for unit test)**

Create `backend/tests/test_ingestion.py`:

```python
import pytest
from pathlib import Path

from app.services.ingestion import get_file_type, SUPPORTED_EXTENSIONS


def test_get_file_type_markdown():
    assert get_file_type(Path("test.md")) == "markdown"


def test_get_file_type_pdf():
    assert get_file_type(Path("doc.pdf")) == "pdf"


def test_get_file_type_code():
    assert get_file_type(Path("main.py")) == "code"
    assert get_file_type(Path("app.tsx")) == "code"


def test_get_file_type_csv():
    assert get_file_type(Path("data.csv")) == "csv"


def test_get_file_type_image():
    assert get_file_type(Path("photo.jpg")) == "image"


def test_get_file_type_unknown():
    assert get_file_type(Path("file.xyz")) == "unknown"


def test_supported_extensions_include_common_types():
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".py" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".json" in SUPPORTED_EXTENSIONS
```

**Step 3: Run tests**

```bash
cd backend
python -m pytest tests/test_ingestion.py -v
```

Expected: All tests PASS.

**Step 4: Commit**

```bash
git add backend/app/services/ingestion.py backend/tests/test_ingestion.py
git commit -m "feat: add ingestion service with multi-format document loading"
```

---

### Task 4: Query Service (RAG Pipeline)

**Files:**
- Create: `backend/app/services/query.py`

**Step 1: Create `backend/app/services/query.py`**

```python
import json
from openai import AsyncOpenAI

from app.config import settings
from app.services.storage import VectorStore, ChatStore
from app.models.schemas import SourceCitation

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the user's personal knowledge base.

When answering:
- Base your answers ONLY on the provided context from the user's documents
- If the context doesn't contain enough information to answer, say so clearly
- Cite which documents your answer comes from
- Be concise and direct

Context from the user's knowledge base:
{context}"""


class QueryService:
    def __init__(self, vector_store: VectorStore, chat_store: ChatStore):
        self.vector_store = vector_store
        self.chat_store = chat_store
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embed_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _get_query_embedding(self, query: str) -> list[float]:
        response = await self.embed_client.embeddings.create(
            model=settings.embedding_model,
            input=query,
        )
        return response.data[0].embedding

    def _format_context(self, results: dict) -> tuple[str, list[SourceCitation]]:
        if not results["documents"] or not results["documents"][0]:
            return "", []

        context_parts = []
        citations = []

        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1 - distance
            context_parts.append(f"[Source {i+1}: {metadata.get('filename', 'unknown')}]\n{doc}\n")
            citations.append(SourceCitation(
                document_name=metadata.get("filename", "unknown"),
                chunk_text=doc[:200] + "..." if len(doc) > 200 else doc,
                score=round(score, 3),
            ))

        return "\n---\n".join(context_parts), citations

    async def query(self, question: str, session_id: str | None = None):
        if session_id is None:
            short_title = question[:50] + "..." if len(question) > 50 else question
            session_id = await self.chat_store.create_session(short_title)

        await self.chat_store.add_message(session_id, "user", question)

        query_embedding = await self._get_query_embedding(question)
        results = self.vector_store.query(query_embedding, top_k=settings.top_k)
        context, citations = self._format_context(results)

        if not context:
            no_context_msg = "I couldn't find any relevant information in your knowledge base to answer this question. Try uploading some documents first."
            await self.chat_store.add_message(session_id, "assistant", no_context_msg)
            yield {"type": "session_id", "data": session_id}
            yield {"type": "content", "data": no_context_msg}
            yield {"type": "sources", "data": []}
            yield {"type": "done", "data": ""}
            return

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]

        session = await self.chat_store.get_session(session_id)
        if session and len(session.messages) > 1:
            history_messages = [
                {"role": m.role, "content": m.content}
                for m in session.messages[:-1]
            ]
            messages = [messages[0]] + history_messages + [messages[-1]]

        yield {"type": "session_id", "data": session_id}
        yield {"type": "sources", "data": [c.model_dump() for c in citations]}

        full_response = ""
        stream = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield {"type": "content", "data": content}

        await self.chat_store.add_message(
            session_id, "assistant", full_response,
            [c.model_dump() for c in citations],
        )
        yield {"type": "done", "data": ""}
```

**Step 2: Commit**

```bash
git add backend/app/services/query.py
git commit -m "feat: add RAG query service with streaming and source citations"
```

---

### Task 5: API Routers

**Files:**
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/routers/ingest.py`
- Create: `backend/app/routers/chat.py`
- Create: `backend/app/routers/documents.py`
- Modify: `backend/app/main.py` (register routers + init services)

**Step 1: Create `backend/app/routers/ingest.py`**

```python
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from app.models.schemas import IngestResponse

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


class DirectoryRequest(BaseModel):
    path: str


class URLRequest(BaseModel):
    url: str


@router.post("/upload", response_model=list[IngestResponse])
async def upload_files(files: list[UploadFile] = File(...)):
    from app.main import get_ingestion_service
    service = get_ingestion_service()

    results = []
    for file in files:
        try:
            content = await file.read()
            result = await service.ingest_uploaded_file(content, file.filename)
            results.append(result)
        except Exception as e:
            results.append(IngestResponse(
                document_id="",
                filename=file.filename or "unknown",
                chunk_count=0,
                status=f"error: {str(e)}",
            ))
    return results


@router.post("/directory", response_model=list[IngestResponse])
async def ingest_directory(request: DirectoryRequest):
    from app.main import get_ingestion_service
    service = get_ingestion_service()

    directory = Path(request.path).expanduser().resolve()
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.path}")
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {request.path}")

    return await service.ingest_directory(directory)


@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: URLRequest):
    from app.main import get_ingestion_service
    service = get_ingestion_service()

    try:
        return await service.ingest_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**Step 2: Create `backend/app/routers/documents.py`**

```python
from fastapi import APIRouter, HTTPException

from app.models.schemas import DocumentMetadata

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentMetadata])
async def list_documents():
    from app.main import get_doc_store, get_vector_store
    doc_store = get_doc_store()
    return await doc_store.list_documents()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    from app.main import get_doc_store, get_vector_store
    doc_store = get_doc_store()
    vector_store = get_vector_store()

    doc = await doc_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store.delete_document(doc_id)
    await doc_store.delete_document(doc_id)
    return {"status": "deleted", "document_id": doc_id}
```

**Step 3: Create `backend/app/routers/chat.py`**

```python
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import QueryRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def chat(request: QueryRequest):
    from app.main import get_query_service
    service = get_query_service()

    async def event_stream():
        async for event in service.query(request.question, request.session_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/history")
async def chat_history():
    from app.main import get_chat_store
    store = get_chat_store()
    return await store.list_sessions()


@router.get("/history/{session_id}")
async def get_session(session_id: str):
    from app.main import get_chat_store
    store = get_chat_store()
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
```

**Step 4: Update `backend/app/main.py` to register routers and init services**

Replace the entire file:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.storage import VectorStore, ChatStore, DocumentStore
from app.services.ingestion import IngestionService
from app.services.query import QueryService
from app.routers import ingest, chat, documents

_vector_store: VectorStore | None = None
_chat_store: ChatStore | None = None
_doc_store: DocumentStore | None = None
_ingestion_service: IngestionService | None = None
_query_service: QueryService | None = None


def get_vector_store() -> VectorStore:
    return _vector_store

def get_chat_store() -> ChatStore:
    return _chat_store

def get_doc_store() -> DocumentStore:
    return _doc_store

def get_ingestion_service() -> IngestionService:
    return _ingestion_service

def get_query_service() -> QueryService:
    return _query_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vector_store, _chat_store, _doc_store, _ingestion_service, _query_service

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    _vector_store = VectorStore()
    _chat_store = ChatStore()
    _doc_store = DocumentStore()

    await _chat_store.init_db()
    await _doc_store.init_db()

    _ingestion_service = IngestionService(_vector_store, _doc_store)
    _query_service = QueryService(_vector_store, _chat_store)

    yield


app = FastAPI(title="Local Knowledge", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

**Step 5: Create empty `backend/app/routers/__init__.py`**

Empty file.

**Step 6: Verify server starts**

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Test: `curl http://localhost:8000/api/health` → `{"status":"ok"}`
Test: `curl http://localhost:8000/api/documents` → `[]`

**Step 7: Commit**

```bash
git add backend/
git commit -m "feat: add API routers for ingest, chat, and documents"
```

---

### Task 6: Frontend Scaffolding

**Files:**
- Create: `frontend/` (via Vite scaffold)
- Modify: `frontend/vite.config.ts` (add proxy)
- Modify: `frontend/tailwind.config.js`
- Create: `frontend/src/api/client.ts`

**Step 1: Scaffold React + Vite + TypeScript project**

```bash
cd /Users/cch102/Desktop/workspace/local-knowledge
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
```

**Step 2: Configure Tailwind**

Update `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

Replace `frontend/src/index.css` with:

```css
@import "tailwindcss";
```

**Step 3: Create `frontend/src/api/client.ts`**

```typescript
const API_BASE = '/api';

export async function uploadFiles(files: File[]): Promise<any[]> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const response = await fetch(`${API_BASE}/ingest/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
  return response.json();
}

export async function ingestDirectory(path: string): Promise<any[]> {
  const response = await fetch(`${API_BASE}/ingest/directory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });

  if (!response.ok) throw new Error(`Ingest failed: ${response.statusText}`);
  return response.json();
}

export async function ingestUrl(url: string): Promise<any> {
  const response = await fetch(`${API_BASE}/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });

  if (!response.ok) throw new Error(`Ingest failed: ${response.statusText}`);
  return response.json();
}

export async function listDocuments(): Promise<any[]> {
  const response = await fetch(`${API_BASE}/documents`);
  if (!response.ok) throw new Error(`Failed to list documents: ${response.statusText}`);
  return response.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${docId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error(`Failed to delete document: ${response.statusText}`);
}

export async function getChatHistory(): Promise<any[]> {
  const response = await fetch(`${API_BASE}/chat/history`);
  if (!response.ok) throw new Error(`Failed to get history: ${response.statusText}`);
  return response.json();
}

export async function getChatSession(sessionId: string): Promise<any> {
  const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
  if (!response.ok) throw new Error(`Failed to get session: ${response.statusText}`);
  return response.json();
}

export interface ChatEvent {
  type: 'session_id' | 'content' | 'sources' | 'done';
  data: any;
}

export async function* streamChat(
  question: string,
  sessionId?: string
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });

  if (!response.ok) throw new Error(`Chat failed: ${response.statusText}`);

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {}
      }
    }
  }
}
```

**Step 4: Verify frontend dev server starts**

```bash
cd frontend
npm run dev
```

Expected: Vite dev server at `http://localhost:5173`

**Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold frontend with React, Vite, TailwindCSS, and API client"
```

---

### Task 7: Frontend Chat Component

**Files:**
- Create: `frontend/src/hooks/useChat.ts`
- Create: `frontend/src/components/Chat.tsx`
- Create: `frontend/src/components/SourceCitation.tsx`

**Step 1: Create `frontend/src/hooks/useChat.ts`**

```typescript
import { useState, useCallback } from 'react';
import { streamChat, type ChatEvent } from '../api/client';

interface Source {
  document_name: string;
  chunk_text: string;
  score: number;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();

  const sendMessage = useCallback(async (question: string) => {
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setIsLoading(true);

    let assistantContent = '';
    let sources: Source[] = [];

    setMessages((prev) => [...prev, { role: 'assistant', content: '', sources: [] }]);

    try {
      for await (const event of streamChat(question, sessionId)) {
        switch (event.type) {
          case 'session_id':
            setSessionId(event.data);
            break;
          case 'content':
            assistantContent += event.data;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: 'assistant',
                content: assistantContent,
                sources,
              };
              return updated;
            });
            break;
          case 'sources':
            sources = event.data;
            break;
          case 'done':
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: 'assistant',
                content: assistantContent,
                sources,
              };
              return updated;
            });
            break;
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'Sorry, something went wrong. Please try again.',
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const clearChat = useCallback(() => {
    setMessages([]);
    setSessionId(undefined);
  }, []);

  const loadSession = useCallback((session: { messages: Message[]; id: string }) => {
    setMessages(session.messages);
    setSessionId(session.id);
  }, []);

  return { messages, isLoading, sessionId, sendMessage, clearChat, loadSession };
}
```

**Step 2: Create `frontend/src/components/SourceCitation.tsx`**

```tsx
interface Source {
  document_name: string;
  chunk_text: string;
  score: number;
}

interface Props {
  sources: Source[];
}

export function SourceCitation({ sources }: Props) {
  if (!sources.length) return null;

  return (
    <div className="mt-3 border-t border-zinc-700 pt-3">
      <p className="text-xs font-medium text-zinc-400 mb-2">Sources</p>
      <div className="flex flex-wrap gap-2">
        {sources.map((source, i) => (
          <div
            key={i}
            className="group relative text-xs bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 cursor-default hover:border-zinc-500 transition-colors"
          >
            <span className="text-zinc-300">{source.document_name}</span>
            <span className="ml-2 text-zinc-500">{Math.round(source.score * 100)}%</span>
            <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block w-80 p-3 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-10">
              <p className="text-zinc-300 text-xs leading-relaxed whitespace-pre-wrap">
                {source.chunk_text}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Step 3: Create `frontend/src/components/Chat.tsx`**

```tsx
import { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { SourceCitation } from './SourceCitation';

interface Props {
  onSessionCreated?: (sessionId: string) => void;
}

export function Chat({ onSessionCreated }: Props) {
  const { messages, isLoading, sessionId, sendMessage, clearChat } = useChat();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (sessionId && onSessionCreated) {
      onSessionCreated(sessionId);
    }
  }, [sessionId, onSessionCreated]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-zinc-500">
            <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
            <p className="text-lg font-medium">Ask your knowledge base</p>
            <p className="text-sm mt-1">Upload documents and ask questions about them</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-zinc-800 text-zinc-100'
            }`}>
              <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
              {msg.role === 'assistant' && msg.sources && (
                <SourceCitation sources={msg.sources} />
              )}
              {msg.role === 'assistant' && isLoading && i === messages.length - 1 && !msg.content && (
                <div className="flex gap-1">
                  <div className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.3s]" />
                  <div className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.15s]" />
                  <div className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce" />
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-zinc-800 p-4">
        <form onSubmit={handleSubmit} className="flex gap-3 max-w-4xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-zinc-100 placeholder-zinc-500 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-xl px-5 py-3 font-medium transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
```

**Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add Chat component with streaming and source citations"
```

---

### Task 8: Frontend Sidebar & File Upload

**Files:**
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/components/FileUpload.tsx`

**Step 1: Create `frontend/src/components/FileUpload.tsx`**

```tsx
import { useState, useCallback } from 'react';
import { uploadFiles, ingestDirectory, ingestUrl } from '../api/client';

interface Props {
  onIngested: () => void;
}

export function FileUpload({ onIngested }: Props) {
  const [isUploading, setIsUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [showDirInput, setShowDirInput] = useState(false);
  const [urlValue, setUrlValue] = useState('');
  const [dirValue, setDirValue] = useState('');
  const [status, setStatus] = useState<string | null>(null);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    setIsUploading(true);
    setStatus(null);
    try {
      const results = await uploadFiles(Array.from(files));
      const succeeded = results.filter((r: any) => r.status === 'success').length;
      const failed = results.length - succeeded;
      setStatus(`Uploaded ${succeeded} file${succeeded !== 1 ? 's' : ''}${failed ? `, ${failed} failed` : ''}`);
      onIngested();
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsUploading(false);
    }
  }, [onIngested]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleUrlSubmit = async () => {
    if (!urlValue.trim()) return;
    setIsUploading(true);
    setStatus(null);
    try {
      await ingestUrl(urlValue.trim());
      setStatus('URL ingested successfully');
      setUrlValue('');
      setShowUrlInput(false);
      onIngested();
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDirSubmit = async () => {
    if (!dirValue.trim()) return;
    setIsUploading(true);
    setStatus(null);
    try {
      const results = await ingestDirectory(dirValue.trim());
      const succeeded = results.filter((r: any) => r.status === 'success').length;
      setStatus(`Ingested ${succeeded} files from directory`);
      setDirValue('');
      setShowDirInput(false);
      onIngested();
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-4 text-center transition-colors cursor-pointer ${
          dragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 hover:border-zinc-500'
        }`}
        onClick={() => {
          const input = document.createElement('input');
          input.type = 'file';
          input.multiple = true;
          input.onchange = (e) => {
            const files = (e.target as HTMLInputElement).files;
            if (files) handleFiles(files);
          };
          input.click();
        }}
      >
        {isUploading ? (
          <div className="text-blue-400 text-sm">Processing...</div>
        ) : (
          <div className="text-zinc-400 text-sm">
            <p className="font-medium">Drop files here</p>
            <p className="text-xs mt-1">or click to browse</p>
          </div>
        )}
      </div>

      {/* URL input */}
      <div className="flex gap-2">
        {showUrlInput ? (
          <>
            <input
              type="text"
              value={urlValue}
              onChange={(e) => setUrlValue(e.target.value)}
              placeholder="https://..."
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              onKeyDown={(e) => e.key === 'Enter' && handleUrlSubmit()}
            />
            <button onClick={handleUrlSubmit} disabled={isUploading}
              className="text-xs bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-3 py-1.5">
              Add
            </button>
            <button onClick={() => setShowUrlInput(false)}
              className="text-xs text-zinc-500 hover:text-zinc-300">
              Cancel
            </button>
          </>
        ) : (
          <button onClick={() => setShowUrlInput(true)}
            className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors">
            + Add URL
          </button>
        )}
      </div>

      {/* Directory input */}
      <div className="flex gap-2">
        {showDirInput ? (
          <>
            <input
              type="text"
              value={dirValue}
              onChange={(e) => setDirValue(e.target.value)}
              placeholder="/path/to/folder"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              onKeyDown={(e) => e.key === 'Enter' && handleDirSubmit()}
            />
            <button onClick={handleDirSubmit} disabled={isUploading}
              className="text-xs bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-3 py-1.5">
              Add
            </button>
            <button onClick={() => setShowDirInput(false)}
              className="text-xs text-zinc-500 hover:text-zinc-300">
              Cancel
            </button>
          </>
        ) : (
          <button onClick={() => setShowDirInput(true)}
            className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors">
            + Add Folder
          </button>
        )}
      </div>

      {status && (
        <p className={`text-xs ${status.startsWith('Error') ? 'text-red-400' : 'text-green-400'}`}>
          {status}
        </p>
      )}
    </div>
  );
}
```

**Step 2: Create `frontend/src/components/Sidebar.tsx`**

```tsx
import { useState, useEffect, useCallback } from 'react';
import { listDocuments, deleteDocument, getChatHistory, getChatSession } from '../api/client';
import { FileUpload } from './FileUpload';

interface Props {
  onSelectSession: (session: any) => void;
  onNewChat: () => void;
  refreshTrigger: number;
}

export function Sidebar({ onSelectSession, onNewChat, refreshTrigger }: Props) {
  const [documents, setDocuments] = useState<any[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'chat' | 'docs'>('chat');

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {}
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const history = await getChatHistory();
      setSessions(history);
    } catch {}
  }, []);

  useEffect(() => {
    refreshDocuments();
    refreshSessions();
  }, [refreshDocuments, refreshSessions, refreshTrigger]);

  const handleDeleteDoc = async (docId: string) => {
    try {
      await deleteDocument(docId);
      refreshDocuments();
    } catch {}
  };

  const handleSelectSession = async (sessionId: string) => {
    try {
      const session = await getChatSession(sessionId);
      onSelectSession(session);
    } catch {}
  };

  const fileTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      markdown: '📝', pdf: '📄', web: '🌐', code: '💻',
      image: '🖼️', csv: '📊', json: '📋', text: '📃',
    };
    return icons[type] || '📁';
  };

  return (
    <div className="w-72 bg-zinc-900 border-r border-zinc-800 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800">
        <h1 className="text-lg font-semibold text-zinc-100">Local Knowledge</h1>
        <button
          onClick={onNewChat}
          className="mt-3 w-full bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
        >
          + New Chat
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-zinc-800">
        <button
          onClick={() => setActiveTab('chat')}
          className={`flex-1 py-2.5 text-sm font-medium transition-colors ${
            activeTab === 'chat' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          Chats
        </button>
        <button
          onClick={() => setActiveTab('docs')}
          className={`flex-1 py-2.5 text-sm font-medium transition-colors ${
            activeTab === 'docs' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          Documents ({documents.length})
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {activeTab === 'chat' ? (
          <div className="space-y-1">
            {sessions.length === 0 && (
              <p className="text-zinc-500 text-sm text-center py-4">No chats yet</p>
            )}
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => handleSelectSession(session.id)}
                className="w-full text-left px-3 py-2 rounded-lg text-sm text-zinc-300 hover:bg-zinc-800 transition-colors truncate"
              >
                {session.title}
              </button>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div key={doc.id} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800/50 group">
                <span>{fileTypeIcon(doc.file_type)}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-zinc-300 truncate">{doc.filename}</p>
                  <p className="text-xs text-zinc-500">{doc.chunk_count} chunks</p>
                </div>
                <button
                  onClick={() => handleDeleteDoc(doc.id)}
                  className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload section */}
      <div className="p-3 border-t border-zinc-800">
        <FileUpload onIngested={refreshDocuments} />
      </div>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: add Sidebar with file upload, document list, and chat history"
```

---

### Task 9: Wire Up the App

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Replace `frontend/src/App.tsx`**

```tsx
import { useState, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { Chat } from './components/Chat';

function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleSessionCreated = useCallback(() => {
    setRefreshTrigger((n) => n + 1);
  }, []);

  const handleNewChat = useCallback(() => {
    window.location.reload();
  }, []);

  const handleSelectSession = useCallback((session: any) => {
    // For now, reload — full state management would be a future enhancement
    window.location.reload();
  }, []);

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      <Sidebar
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        refreshTrigger={refreshTrigger}
      />
      <main className="flex-1 flex flex-col">
        <Chat onSessionCreated={handleSessionCreated} />
      </main>
    </div>
  );
}

export default App;
```

**Step 2: Clean up default files**

Delete `frontend/src/App.css` and any Vite default content. Ensure `frontend/src/main.tsx` imports `./index.css`.

**Step 3: Verify full stack runs**

Terminal 1:
```bash
cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000
```

Terminal 2:
```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`. The app should show the sidebar + chat interface. Upload a file and ask a question.

**Step 4: Commit**

```bash
git add frontend/
git commit -m "feat: wire up App with Sidebar and Chat components"
```

---

### Task 10: README and Final Polish

**Files:**
- Create: `README.md`

**Step 1: Create `README.md`**

```markdown
# Local Knowledge

A personal knowledge base with RAG-powered chat. Ingest your documents (markdown, PDFs, web pages, code, images, CSV/JSON) and ask questions through a chat interface that synthesizes answers with source citations.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- An OpenAI API key

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Usage

1. Upload documents via the sidebar (drag-and-drop, URL, or local folder path)
2. Ask questions in the chat — answers are synthesized from your documents with source citations
3. Browse and manage your documents in the Documents tab
4. Chat history is preserved in the sidebar

## Supported File Types

| Type | Extensions |
|------|-----------|
| Markdown/Text | .md, .txt |
| PDF | .pdf |
| Web | .html, .htm (or paste a URL) |
| Code | .py, .js, .ts, .tsx, .jsx, .java, .go, .rs, .c, .cpp, .h |
| Images | .png, .jpg, .jpeg (OCR) |
| Data | .csv, .json |

## Architecture

- **Backend:** Python FastAPI + LlamaIndex + ChromaDB
- **Frontend:** React + Vite + TailwindCSS
- **LLM:** OpenAI GPT-4o (configurable)
- **Embeddings:** OpenAI text-embedding-3-small
- **Storage:** Local filesystem + ChromaDB (on disk) + SQLite
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions"
```
