# Personal Knowledge Search — Design Document

**Date:** 2026-04-09
**Status:** Approved

## Goal

A personal knowledge base with a RAG chat interface. Users ingest mixed content (markdown, PDFs, web pages, code, images, CSV/JSON), and ask natural-language questions through a web UI that synthesizes answers grounded in their data with source citations.

## Architecture

```
React UI (Chat + Upload)  ⟷  FastAPI Backend
                                  │
                          ┌───────┴───────┐
                          │               │
                    Ingestion        Query Engine
                    Pipeline         (LlamaIndex)
                    (LlamaIndex)          │
                          │               │
                          ▼               ▼
                      ChromaDB (vectors on disk)
                          │               │
                          ▼               ▼
                      OpenAI API (embeddings + chat)
```

### Data Flow

**Ingest:** Files uploaded or directory scanned → LlamaIndex readers parse each type → text chunked (1024 tokens, 128 overlap) → embedded via OpenAI `text-embedding-3-small` → stored in ChromaDB with metadata.

**Query:** User question → embedded → top-k chunks retrieved from ChromaDB → chunks + question sent to LLM → streamed answer returned with source citations.

### Storage

- `./data/uploads/` — ingested files
- `./data/chroma/` — ChromaDB persistence (local, no server needed)
- Chat history stored in a lightweight SQLite database at `./data/chat.db`

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/ingest/upload` | POST | Upload files (multipart form) |
| `/api/ingest/directory` | POST | Ingest all files from a local directory path |
| `/api/documents` | GET | List ingested documents with metadata |
| `/api/documents/{id}` | DELETE | Remove a document and its vectors |
| `/api/chat` | POST | Send a question, get a streamed answer (SSE) |
| `/api/chat/history` | GET | Retrieve past conversations |

## File Type Support

| File Type | LlamaIndex Reader | Notes |
|---|---|---|
| Markdown / Text | `MarkdownReader` | Preserves headers as metadata |
| PDF | `PDFReader` | Text extraction; OCR fallback for scanned pages |
| Web pages | `BeautifulSoupWebReader` | Pass URLs, extracts readable text |
| Code | `SimpleDirectoryReader` | Language-aware chunking |
| Images | `ImageReader` (OCR) | pytesseract or vision model |
| CSV/JSON | `PandasCSVReader` / `JSONReader` | Structured data → searchable text |

Metadata per document: file name, file type, ingestion timestamp, source path/URL, chunk count.

## Frontend

Single-page React app with sidebar + main chat area.

**Layout:**
- Sidebar: collections (tag-based grouping), recent chats, upload button
- Main area: chat messages with streaming, source citations, input box

**Features:**
- Streaming answers (token-by-token via SSE)
- Source citations with document name, chunk snippet, relevance score
- Drag-and-drop file upload with progress/status
- Document library (browse, search, delete)
- Collections (simple tag-based folder structure)
- Chat history (persisted, browsable from sidebar)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, uvicorn |
| RAG pipeline | LlamaIndex |
| Vector store | ChromaDB (local persistence) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o (configurable) |
| Frontend | React 18, Vite, TailwindCSS |
| Chat DB | SQLite |

## Project Structure

```
local-knowledge/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/
│   │   │   ├── ingest.py
│   │   │   ├── chat.py
│   │   │   └── documents.py
│   │   ├── services/
│   │   │   ├── ingestion.py
│   │   │   ├── query.py
│   │   │   └── storage.py
│   │   └── models/
│   │       └── schemas.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Chat.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── SourceCitation.tsx
│   │   │   └── FileUpload.tsx
│   │   ├── hooks/
│   │   │   └── useChat.ts
│   │   └── api/
│   │       └── client.ts
│   ├── package.json
│   └── vite.config.ts
├── data/
│   ├── uploads/
│   └── chroma/
└── docs/
    └── plans/
```

## Error Handling

- Failed ingestion: clear per-file errors (unsupported format, parse failure, OCR failure)
- Chat errors: graceful fallback message ("couldn't retrieve relevant information")
- All API errors: structured JSON responses with error codes

## Testing

- Backend: unit tests for ingestion/query services, API integration tests
- Frontend: component tests for chat behavior, upload flow
