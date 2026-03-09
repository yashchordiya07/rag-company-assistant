"""
app/main.py
─────────────────────────────────────────────────────────────
FastAPI application entry point.

Responsibilities
----------------
* Create the FastAPI app with lifespan context manager
* Initialise the RAG pipeline (singleton) on startup
* Mount the API router
* Configure CORS middleware
* Expose a Uvicorn-compatible ``app`` object
* Provide the ``get_app_pipeline()`` accessor used by route dependencies

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api_routes import router
from app.config import get_settings
from rag.rag_pipeline import RAGPipeline
from utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# ─── Global pipeline state ───────────────────────────────────
_pipeline: RAGPipeline | None = None


def get_app_pipeline() -> RAGPipeline | None:
    """Return the application-level RAGPipeline singleton."""
    return _pipeline


# ─── Lifespan ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.

    Startup:
        - Loads configuration
        - Initialises the RAG pipeline (loads persisted FAISS index if present)
        - Seeds sample documents if the vector store is empty (dev mode)

    Shutdown:
        - Saves the FAISS index to disk
    """
    global _pipeline
    settings = get_settings()
    setup_logging(str(settings.log_level), settings.log_file)

    log.info("=" * 60)
    log.info("Private Knowledge AI Assistant — starting up")
    log.info(f"LLM provider : {settings.llm_provider}")
    log.info(f"Embedding    : {settings.embedding_model}")
    log.info(f"Vector store : {settings.vector_store_path}")
    log.info("=" * 60)

    try:
        _pipeline = RAGPipeline.from_config(settings)
        log.info(
            f"Pipeline ready. Indexed chunks: "
            f"{_pipeline.vector_store.num_chunks}"
        )
    except Exception as exc:
        log.error(f"Pipeline init failed: {exc}")
        raise

    yield  # ←── server is live here

    # Shutdown
    log.info("Shutting down — saving vector store…")
    if _pipeline:
        _pipeline.vector_store.save()
    log.info("Goodbye 👋")


# ─── App factory ─────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Private Knowledge AI Assistant",
        description=(
            "Enterprise RAG system that answers questions from internal "
            "documents using retrieval-augmented generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        return response

    # ── Global exception handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check logs for details."},
        )

    # ── Include routes ────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


# ─── Dev runner ──────────────────────────────────────────────
if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
