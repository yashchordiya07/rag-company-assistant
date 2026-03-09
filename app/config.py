"""
app/config.py
─────────────────────────────────────────────────────────────
Central configuration module for the RAG system.

Uses Pydantic Settings to load values from environment variables
or a .env file, providing type-validated, auto-documented config
across every component in the application.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── LLM ─────────────────────────────────────────────────
    llm_provider: Literal["openai", "ollama", "huggingface"] = Field(
        default="openai", description="LLM backend to use"
    )
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI chat model")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    ollama_model: str = Field(default="llama3.2", description="Ollama model name")
    huggingface_api_token: str = Field(default="", description="HuggingFace API token")
    huggingface_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HuggingFace model ID",
    )

    # ─── Embeddings ──────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="SentenceTransformer model name"
    )
    embedding_device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device for embedding inference"
    )

    # ─── Vector Store ────────────────────────────────────────
    vector_store_path: Path = Field(
        default=Path("./data/vector_store"),
        description="Directory for persisted FAISS index",
    )
    faiss_index_name: str = Field(
        default="company_knowledge", description="FAISS index file prefix"
    )

    # ─── Chunking ────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Token size per chunk", ge=64)
    chunk_overlap: int = Field(default=64, description="Token overlap between chunks", ge=0)
    min_chunk_length: int = Field(
        default=50, description="Minimum characters to keep a chunk", ge=10
    )

    # ─── Retrieval ───────────────────────────────────────────
    top_k_retrieval: int = Field(
        default=10, description="Candidates retrieved before re-ranking", ge=1
    )
    top_k_final: int = Field(
        default=5, description="Final results returned to the user", ge=1
    )
    similarity_threshold: float = Field(
        default=0.3, description="Minimum cosine similarity to include a chunk", ge=0.0, le=1.0
    )

    # ─── Re-ranking ──────────────────────────────────────────
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking",
    )
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder re-ranking")

    # ─── Hybrid Search ───────────────────────────────────────
    enable_hybrid_search: bool = Field(
        default=True, description="Combine BM25 + semantic search"
    )
    bm25_weight: float = Field(default=0.3, description="BM25 score weight", ge=0.0, le=1.0)
    semantic_weight: float = Field(
        default=0.7, description="Semantic score weight", ge=0.0, le=1.0
    )

    # ─── Caching ─────────────────────────────────────────────
    enable_cache: bool = Field(default=False, description="Enable Redis query caching")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    cache_ttl_seconds: int = Field(
        default=3600, description="Cache entry time-to-live in seconds", ge=60
    )

    # ─── Conversation Memory ─────────────────────────────────
    max_history_turns: int = Field(
        default=10, description="Max conversation turns kept in memory", ge=1
    )
    enable_query_rewriting: bool = Field(
        default=True, description="Rewrite queries using conversation context"
    )

    # ─── API ─────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI bind port")
    api_reload: bool = Field(default=True, description="Uvicorn auto-reload in dev")

    # ─── UI ──────────────────────────────────────────────────
    ui_port: int = Field(default=8501, description="Streamlit port")
    api_base_url: str = Field(
        default="http://localhost:8000", description="Backend URL used by Streamlit"
    )

    # ─── Logging ─────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log verbosity"
    )
    log_file: Path = Field(default=Path("./logs/rag_system.log"), description="Log file path")

    # ─── Upload ──────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=50, description="Max document upload size in MB")
    allowed_extensions: list[str] = Field(
        default=["pdf", "txt", "docx", "md"],
        description="Allowed document extensions",
    )

    # ─── Validators ──────────────────────────────────────────
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    @field_validator("vector_store_path", "log_file", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    # ─── Derived Properties ───────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_store_path / self.faiss_index_name

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        Path("./data/sample_docs").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
