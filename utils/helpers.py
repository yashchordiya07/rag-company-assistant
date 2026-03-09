"""
utils/helpers.py
─────────────────────────────────────────────────────────────
General-purpose utility functions used across the RAG system.

Covers:
- Text cleaning & normalisation
- Token counting (tiktoken)
- File-type detection
- Unique ID generation
- Similarity score formatting
- Safe JSON loading
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import tiktoken

from utils.logger import get_logger

log = get_logger(__name__)

# ─── Tokeniser (cached) ──────────────────────────────────────
_TOKENISER: tiktoken.Encoding | None = None


def get_tokeniser() -> tiktoken.Encoding:
    """Return a cached cl100k_base tokeniser."""
    global _TOKENISER
    if _TOKENISER is None:
        _TOKENISER = tiktoken.get_encoding("cl100k_base")
    return _TOKENISER


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    return len(get_tokeniser().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate *text* so it fits within *max_tokens*.

    Returns the original string if it is already short enough.
    """
    enc = get_tokeniser()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ─── Text cleaning ───────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise raw extracted text from documents.

    Steps
    -----
    1. Collapse multiple blank lines → single blank line
    2. Strip leading / trailing whitespace on each line
    3. Remove non-printable control characters (except newlines/tabs)
    4. Collapse multiple spaces → single space
    """
    # Remove null bytes and other non-printable chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    # Collapse 3+ consecutive blank lines → 2
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) to a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ─── File helpers ────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {"pdf", "txt", "docx", "md"}


def get_file_extension(path: str | Path) -> str:
    """Return lowercase extension without the leading dot."""
    return Path(path).suffix.lstrip(".").lower()


def is_supported_file(path: str | Path) -> bool:
    """Return True if the file extension is in the supported set."""
    return get_file_extension(path) in SUPPORTED_EXTENSIONS


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file (for deduplication)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    """Return an MD5 hex digest of a string (fast, for cache keys)."""
    return hashlib.md5(text.encode()).hexdigest()


# ─── ID generation ───────────────────────────────────────────
def new_uuid() -> str:
    """Return a new random UUID string."""
    return str(uuid.uuid4())


def chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from document ID and index."""
    return f"{doc_id}::chunk_{chunk_index:04d}"


# ─── Score formatting ────────────────────────────────────────
def format_score(score: float, decimals: int = 4) -> str:
    """Format a similarity score as a readable percentage string."""
    pct = score * 100
    return f"{pct:.{decimals - 2}f}%"


def scores_to_percentile(scores: list[float]) -> list[float]:
    """
    Convert a list of raw scores to [0, 1] range using min-max scaling.
    Returns the original list if all scores are equal.
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


# ─── JSON helpers ────────────────────────────────────────────
def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any error."""
    try:
        return json.loads(text)
    except Exception as exc:
        log.warning(f"JSON parse failed: {exc}")
        return default


def pretty_json(obj: Any) -> str:
    """Serialise *obj* to an indented JSON string."""
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


# ─── Prompt helpers ──────────────────────────────────────────
def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunks into a numbered context block
    suitable for inclusion in a prompt.

    Each chunk dict is expected to have at least:
    - ``text``   : str — the chunk content
    - ``source`` : str — document name / path
    - ``page``   : int | None — page number (optional)
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "")
        page_str = f", page {page}" if page else ""
        parts.append(
            f"[{i}] Source: {source}{page_str}\n"
            f"{chunk.get('text', '').strip()}"
        )
    return "\n\n".join(parts)


def extract_citations(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract citation references from an LLM answer.

    Looks for patterns like [1], [2], [1,2] and maps them to the
    corresponding chunks in *chunks*.
    """
    cited_indices: set[int] = set()
    for match in re.finditer(r"\[(\d+(?:,\s*\d+)*)\]", answer):
        for num in match.group(1).split(","):
            idx = int(num.strip()) - 1  # convert to 0-based
            if 0 <= idx < len(chunks):
                cited_indices.add(idx)

    citations: list[dict[str, Any]] = []
    for idx in sorted(cited_indices):
        chunk = chunks[idx]
        citations.append(
            {
                "index": idx + 1,
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page"),
                "preview": chunk.get("text", "")[:200].strip() + "…",
            }
        )
    return citations
