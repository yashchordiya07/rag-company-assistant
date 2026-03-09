"""
app/api_routes.py
─────────────────────────────────────────────────────────────
FastAPI route definitions for the RAG system backend.

Endpoints
---------
GET  /health            — Liveness probe
GET  /stats             — System statistics
POST /upload_document   — Ingest a document file
POST /query             — Answer a question
DELETE /memory          — Clear conversation memory
DELETE /index           — Wipe the vector store (admin)

All request/response schemas are defined as Pydantic v2 models
for automatic OpenAPI documentation and validation.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from rag.rag_pipeline import RAGPipeline
from utils.helpers import get_file_extension
from utils.logger import get_logger, set_request_id

log = get_logger(__name__)
router = APIRouter()

# ─── Shared pipeline state ───────────────────────────────────
# The pipeline singleton is initialised in main.py's lifespan handler
# and stored on the app's state. We access it via a dependency below.

def get_pipeline(settings: Annotated[Settings, Depends(get_settings)]) -> RAGPipeline:
    """FastAPI dependency — returns the global RAGPipeline instance."""
    from app.main import get_app_pipeline  # avoid circular import
    pipeline = get_app_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised yet. Please wait.",
        )
    return pipeline


# ─── Request / Response Schemas ──────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: float = Field(default_factory=time.time)


class StatsResponse(BaseModel):
    pipeline_stats: dict[str, Any]
    settings: dict[str, Any]


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's natural language question",
        examples=["What is the company leave policy?"],
    )
    session_id: str = Field(
        default="default",
        description="Conversation session ID for memory isolation",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max results to retrieve",
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming (not yet supported via REST, use WebSocket)",
    )


class CitationSchema(BaseModel):
    index: int
    source: str
    page: int | None
    preview: str


class ChunkSchema(BaseModel):
    chunk_id: str
    source: str
    file_name: str
    page_number: int | None
    text: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    request_id: str
    question: str
    rewritten_question: str
    answer: str
    citations: list[CitationSchema]
    retrieved_chunks: list[ChunkSchema]
    latency_ms: float
    from_cache: bool


class UploadResponse(BaseModel):
    request_id: str
    doc_id: str
    file_name: str
    pages: int
    chunks: int
    chunks_indexed: int
    char_count: int
    latency_ms: float
    skipped: bool
    message: str


class ClearMemoryResponse(BaseModel):
    message: str = "Conversation memory cleared"


class ClearIndexResponse(BaseModel):
    message: str = "Vector store cleared"


# ─── Endpoints ───────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe",
)
async def health_check() -> HealthResponse:
    """Returns 200 OK if the server is running."""
    return HealthResponse()


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["System"],
    summary="Pipeline and configuration statistics",
)
async def get_stats(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StatsResponse:
    """Returns runtime statistics and configuration summary."""
    return StatsResponse(
        pipeline_stats=pipeline.stats(),
        settings={
            "llm_provider": settings.llm_provider,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "top_k_retrieval": settings.top_k_retrieval,
            "top_k_final": settings.top_k_final,
            "reranking_enabled": settings.enable_reranking,
            "hybrid_search_enabled": settings.enable_hybrid_search,
            "cache_enabled": settings.enable_cache,
        },
    )


@router.post(
    "/upload_document",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and ingest a document",
)
async def upload_document(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(..., description="PDF, DOCX, TXT, or MD file"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> UploadResponse:
    """
    Upload a document to the knowledge base.

    The document is loaded, chunked, embedded, and indexed
    into the FAISS vector store. Subsequent queries will be
    able to retrieve information from this document.
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    # ── Validate file ────────────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    ext = get_file_extension(file.filename)
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '.{ext}' not supported. "
                f"Allowed: {settings.allowed_extensions}"
            ),
        )

    # ── Read file bytes ──────────────────────────────────────
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Max: {settings.max_upload_size_mb} MB"
            ),
        )

    log.info(
        f"Document upload received",
        request_id=request_id,
        filename=file.filename,
        size_mb=f"{size_mb:.2f}",
    )

    # ── Ingest ───────────────────────────────────────────────
    try:
        stats = pipeline.ingest_document(
            path=None,
            data=data,
            file_name=file.filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        log.error(f"Ingestion failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    message = (
        "Document already indexed (skipped)."
        if stats["skipped"]
        else f"Document ingested successfully: {stats['chunks']} chunks indexed."
    )

    return UploadResponse(
        request_id=request_id,
        doc_id=stats["doc_id"],
        file_name=stats["file_name"],
        pages=stats["pages"],
        chunks=stats["chunks"],
        chunks_indexed=stats["chunks_indexed"],
        char_count=stats["char_count"],
        latency_ms=stats["latency_ms"],
        skipped=stats["skipped"],
        message=message,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Ask a question against the knowledge base",
)
async def query_knowledge_base(
    request: QueryRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> QueryResponse:
    """
    Submit a natural language question and receive an AI-generated
    answer with citations from the indexed documents.
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    if pipeline.vector_store.num_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    log.info(
        "Query received",
        request_id=request_id,
        question=request.question[:80],
        session=request.session_id,
    )

    try:
        # Temporarily override top_k if caller specifies
        original_k = pipeline.top_k_final
        if request.top_k != original_k:
            pipeline.top_k_final = request.top_k

        response = await pipeline.aquery(
            question=request.question,
            session_id=request.session_id,
        )

        pipeline.top_k_final = original_k
    except Exception as exc:
        log.error(f"Query failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {exc}",
        )

    return QueryResponse(
        request_id=request_id,
        question=response.question,
        rewritten_question=response.rewritten_question,
        answer=response.answer,
        citations=[CitationSchema(**c) for c in response.citations],
        retrieved_chunks=[
            ChunkSchema(
                chunk_id=c.get("chunk_id", ""),
                source=c.get("source", ""),
                file_name=c.get("file_name", ""),
                page_number=c.get("page_number"),
                text=c.get("text", ""),
                score=c.get("score", 0.0),
                rank=c.get("rank", 0),
            )
            for c in response.retrieved_chunks
        ],
        latency_ms=response.latency_ms,
        from_cache=response.from_cache,
    )


@router.delete(
    "/memory",
    response_model=ClearMemoryResponse,
    tags=["Conversation"],
    summary="Clear conversation memory",
)
async def clear_memory(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> ClearMemoryResponse:
    """Clear the stored conversation history."""
    pipeline.clear_memory()
    return ClearMemoryResponse()


@router.delete(
    "/index",
    response_model=ClearIndexResponse,
    tags=["Admin"],
    summary="Wipe the entire vector store (destructive!)",
)
async def clear_index(
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> ClearIndexResponse:
    """
    ⚠️  Permanently deletes all indexed documents.
    Use with caution — this cannot be undone.
    """
    pipeline.vector_store.clear()
    pipeline._all_chunks = []
    pipeline._retriever = None
    pipeline._retriever_needs_rebuild = True
    log.warning("Vector store cleared via API")
    return ClearIndexResponse()
