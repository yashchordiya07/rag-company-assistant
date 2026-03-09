# 🧠 Private Knowledge AI Assistant

> **Enterprise-grade Retrieval-Augmented Generation (RAG) system**  
> Answer questions from internal company documents with citations, hybrid search, and cross-encoder re-ranking.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Example Queries](#-example-queries)
- [Advanced Features](#-advanced-features)

---

## 🎯 Problem Statement

Companies store thousands of internal documents (PDFs, reports, policies, meeting notes).
Employees waste hours searching through these documents manually.

This system lets employees **ask questions in plain English** and receive **AI-generated answers with source citations** drawn from your private document library — without any data leaving your infrastructure.

---

## 🏗️ Architecture

```
Documents (PDF / DOCX / TXT / MD)
        │
        ▼
┌─────────────────────┐
│   Document Loader   │  pypdf · python-docx · UTF-8 text
│   (rag/document_    │
│    loader.py)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Text Chunker      │  Token-aware chunking (tiktoken)
│   (rag/chunking.py) │  Sentence-boundary aware · Overlap
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Embedding Model   │  SentenceTransformers
│   (rag/embeddings   │  all-MiniLM-L6-v2 · L2-normalised
│    .py)             │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   FAISS Vector DB   │  IndexFlatIP · Persistent storage
│   (rag/vector_store │  Deduplication · Backup
│    .py)             │
└────────┬────────────┘
         │ ┌────────────────────┐
         │ │    BM25 Index      │  rank-bm25 · Keyword matching
         │ └────────┬───────────┘
         │          │
         ▼          ▼
┌─────────────────────┐
│  Hybrid Retriever   │  RRF Fusion · Semantic + Lexical
│  (rag/retriever.py) │  Query Rewriting · Conversation memory
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Cross-Encoder      │  ms-marco-MiniLM-L-6-v2
│  Re-Ranker          │  Scores (query, passage) jointly
│  (rag/reranker.py)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   LLM (Prompt +     │  OpenAI · Ollama · HuggingFace
│   Citation Gen)     │  Streaming · Retry logic
│  (models/llm_model  │
│   .py)              │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐    ┌──────────────────────┐
│   FastAPI Backend   │◄───│  Streamlit Chat UI   │
│   (app/main.py)     │    │  (ui/streamlit_app   │
│   REST API          │    │   .py)               │
└─────────────────────┘    └──────────────────────┘
```

---

## ✨ Features

### Core
- ✅ **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- ✅ **Intelligent chunking** — Token-aware with sentence boundary detection
- ✅ **Hybrid search** — FAISS semantic + BM25 keyword (RRF fusion)
- ✅ **Cross-encoder re-ranking** — Dramatically improves result quality
- ✅ **Citation generation** — Every claim traced to a source [1], [2]
- ✅ **Multi-document reasoning** — Combines information across files

### Advanced
- ✅ **Query rewriting** — Handles pronouns and follow-up questions
- ✅ **Conversation memory** — Maintains context across turns
- ✅ **Streaming responses** — Token-by-token output supported
- ✅ **In-process caching** — LRU cache for repeated queries
- ✅ **Document deduplication** — SHA-256 hash prevents re-indexing
- ✅ **Persistent FAISS index** — Survives server restarts
- ✅ **Multiple LLM backends** — OpenAI, Ollama (local), HuggingFace

---

## 📁 Project Structure

```
rag-company-assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, lifespan, middleware
│   ├── config.py         # Pydantic Settings (env vars)
│   └── api_routes.py     # REST endpoints
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py # PDF/DOCX/TXT/MD ingestion
│   ├── chunking.py        # Token-aware text chunking
│   ├── embeddings.py      # SentenceTransformers wrapper
│   ├── vector_store.py    # FAISS index + metadata
│   ├── retriever.py       # Hybrid BM25 + semantic retriever
│   ├── reranker.py        # Cross-encoder re-ranking
│   └── rag_pipeline.py    # End-to-end pipeline orchestration
│
├── models/
│   ├── __init__.py
│   └── llm_model.py      # OpenAI / Ollama / HuggingFace LLMs
│
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py  # Chat interface
│
├── utils/
│   ├── __init__.py
│   ├── logger.py         # Loguru-based structured logging
│   └── helpers.py        # Text cleaning, token counting, etc.
│
├── data/
│   ├── sample_docs/      # Drop your documents here
│   │   ├── hr_policy.md
│   │   └── it_security_policy.md
│   └── vector_store/     # Auto-created; FAISS index lives here
│
├── notebooks/
│   └── rag_experiments.ipynb
│
├── logs/                 # Auto-created
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- 4 GB RAM minimum (8 GB recommended for re-ranking)
- An OpenAI API key **or** a local Ollama installation

### Step 1: Clone & enter the project

```bash
git clone https://github.com/your-org/rag-company-assistant.git
cd rag-company-assistant
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads ~400 MB of model weights (embedding + reranker).
> They are cached in `~/.cache/huggingface/` after the first download.

### Step 4: Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
# For OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# For local Ollama (no API key needed)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2
```

---

## ⚙️ Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` \| `ollama` \| `huggingface` |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Candidates before re-ranking |
| `TOP_K_FINAL` | `5` | Final results to LLM |
| `ENABLE_RERANKING` | `true` | Cross-encoder re-ranking |
| `ENABLE_HYBRID_SEARCH` | `true` | BM25 + semantic fusion |
| `ENABLE_CACHE` | `false` | In-memory query cache |

---

## ▶️ Running the Application

### Option A: Backend + UI (recommended)

**Terminal 1 — Start the API server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Start the Streamlit UI:**
```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

### Option B: API only (headless / programmatic)

```bash
uvicorn app.main:app --port 8000
```

Access the interactive API docs at **http://localhost:8000/docs**.

### Option C: Use the pipeline directly in Python

```python
from app.config import get_settings
from rag.rag_pipeline import RAGPipeline

settings = get_settings()
pipeline = RAGPipeline.from_config(settings)

# Ingest a document
pipeline.ingest_document("data/sample_docs/hr_policy.md")

# Ask a question
response = pipeline.query("What is the maternity leave policy?")
print(response.answer)
for cit in response.citations:
    print(f"[{cit['index']}] {cit['source']}")
```

---

## 📡 API Reference

### `GET /api/v1/health`
Liveness probe.
```json
{"status": "ok", "version": "1.0.0", "timestamp": 1735689600}
```

### `GET /api/v1/stats`
System statistics (chunks indexed, config summary).

### `POST /api/v1/upload_document`
Upload a document file (multipart/form-data).

**Request:** `file=@hr_policy.pdf`

**Response:**
```json
{
  "doc_id": "uuid-...",
  "file_name": "hr_policy.pdf",
  "pages": 12,
  "chunks": 48,
  "chunks_indexed": 48,
  "latency_ms": 2340.1
}
```

### `POST /api/v1/query`
Ask a question.

**Request:**
```json
{
  "question": "How many vacation days do employees get?",
  "session_id": "user-123",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Full-time employees are entitled to 20 paid vacation days [1]...",
  "citations": [
    {"index": 1, "source": "hr_policy.md — page 1", "page": 1, "preview": "..."}
  ],
  "retrieved_chunks": [...],
  "latency_ms": 1823.4
}
```

### `DELETE /api/v1/memory`
Clear conversation history.

### `DELETE /api/v1/index`
⚠️ Wipe all indexed documents.

---

## 💬 Example Queries

After uploading the sample documents, try:

```
What is the company leave policy?
How many vacation days are allowed per year?
What is the meal allowance for dinner?
How do I request a salary advance?
What happens if I lose my company laptop?
Can I use personal devices for work?
What is the hotel expense limit when travelling internationally?
How long is paternity leave?
What are the password requirements?
What rating do I need for a salary increase?
```

---

## 🔬 Advanced Features

### Conversation Memory
The system remembers previous questions in the session. Try:
```
Q: What is the leave policy?
Q: How many days can I carry over?  ← system understands "days" refers to vacation days
Q: And what about sick leave?
```

### Query Rewriting
Ambiguous pronouns are automatically resolved:
- "How many does she get?" → "How many vacation days does an employee get?"

### Hybrid Search
Combines semantic similarity (FAISS) with keyword matching (BM25) using
Reciprocal Rank Fusion. Both signals improve retrieval recall.

### Cross-Encoder Re-Ranking
After initial retrieval, a cross-encoder scores each `(query, chunk)` pair
jointly — far more accurate than bi-encoder similarity alone.

### Streaming (CLI)
```python
for token in pipeline.llm.stream(prompt):
    print(token, end="", flush=True)
```

---

## 🧪 Testing

```bash
# Run the experiment notebook
jupyter notebook notebooks/rag_experiments.ipynb

# Or run a quick smoke test
python -c "
from rag.document_loader import DocumentLoader
from rag.chunking import TokenChunker
loader = DocumentLoader()
doc = loader.load('data/sample_docs/hr_policy.md')
chunker = TokenChunker()
chunks = chunker.chunk_document(doc)
print(f'OK: {len(chunks)} chunks from {doc.file_name}')
"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Format code: `black . && ruff check .`
4. Commit: `git commit -m "feat: add my feature"`
5. Open a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using LangChain, FAISS, SentenceTransformers, FastAPI, and Streamlit.*
