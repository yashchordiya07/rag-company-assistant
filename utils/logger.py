"""
utils/logger.py
─────────────────────────────────────────────────────────────
Centralised logging configuration using Loguru.

Features
--------
* Console sink (coloured, human-readable)
* Rotating file sink (JSON-structured for log aggregators)
* Per-module named loggers
* Context variables (request_id, session_id) via `contextvars`
* Convenience helpers: `log_performance`, `log_retrieval_event`
"""

from __future__ import annotations

import sys
import time
import functools
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Generator

from loguru import logger

# ─── Context variables (carry trace IDs across async calls) ──
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def set_session_id(session_id: str) -> None:
    _session_id_var.set(session_id)


# ─── Loguru filter that injects context vars ─────────────────
def _context_filter(record: dict[str, Any]) -> bool:
    record["extra"]["request_id"] = _request_id_var.get()
    record["extra"]["session_id"] = _session_id_var.get()
    return True


# ─── Setup ───────────────────────────────────────────────────
def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure Loguru sinks.

    Parameters
    ----------
    log_level : str
        Minimum log level (DEBUG | INFO | WARNING | ERROR | CRITICAL).
    log_file : Path | None
        Optional path for a rotating JSON log file.
    """
    logger.remove()  # Remove default handler

    # ── Console sink ─────────────────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "[req={extra[request_id]}] "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        filter=_context_filter,
    )

    # ── File sink (rotating, JSON) ────────────────────────────
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | "
            "req={extra[request_id]} | sess={extra[session_id]} | {message}"
        )
        logger.add(
            str(log_file),
            format=file_format,
            level=log_level,
            rotation="50 MB",
            retention="30 days",
            compression="gz",
            serialize=False,
            filter=_context_filter,
        )

    logger.info("Logging initialised", level=log_level, file=str(log_file))


def get_logger(name: str):
    """Return a logger bound to a module name."""
    return logger.bind(name=name)


# ─── Performance decorator ───────────────────────────────────
def log_performance(func: Callable | None = None, *, threshold_ms: float = 0.0):
    """
    Decorator that logs function execution time.

    Usage
    -----
    @log_performance
    def my_func(): ...

    @log_performance(threshold_ms=200)
    async def my_async_func(): ...
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                if elapsed >= threshold_ms:
                    logger.debug(
                        f"⏱  {f.__qualname__} completed in {elapsed:.1f} ms"
                    )
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    f"💥 {f.__qualname__} failed after {elapsed:.1f} ms: {exc}"
                )
                raise

        return wrapper

    return decorator(func) if func is not None else decorator


@contextmanager
def log_block(label: str) -> Generator[None, None, None]:
    """Context manager to time and log a code block."""
    start = time.perf_counter()
    logger.debug(f"▶ {label} — started")
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"◀ {label} — done in {elapsed:.1f} ms")


# ─── Domain-specific helpers ─────────────────────────────────
def log_retrieval_event(
    query: str,
    num_candidates: int,
    num_final: int,
    top_score: float,
    latency_ms: float,
) -> None:
    """Emit a structured INFO log for every retrieval call."""
    logger.info(
        "🔍 Retrieval",
        query_preview=query[:80],
        candidates=num_candidates,
        final=num_final,
        top_score=f"{top_score:.4f}",
        latency_ms=f"{latency_ms:.1f}",
    )


def log_llm_call(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> None:
    """Emit a structured INFO log for every LLM call."""
    logger.info(
        "🤖 LLM call",
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=f"{latency_ms:.1f}",
    )
