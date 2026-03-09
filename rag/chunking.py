"""
rag/chunking.py
─────────────────────────────────────────────────────────────
Text chunking strategies for the RAG pipeline.

Strategies
----------
1. ``TokenChunker``      — Split by token count (tiktoken), respects sentence
                           boundaries.  Used as the primary strategy.
2. ``SemanticChunker``   — Experimental: split on topic shifts detected by
                           embedding cosine distance (not used in default pipeline
                           but available for advanced setups).

Each ``Chunk`` dataclass carries:
    - The chunk text
    - Source document id / name / page reference
    - Character / token offsets within the original document
    - A unique chunk id
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import nltk

from rag.document_loader import Document, PageContent
from utils.helpers import chunk_id, count_tokens, get_tokeniser
from utils.logger import get_logger

log = get_logger(__name__)

# Download NLTK sentence tokeniser data (only on first run)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


# ─── Domain model ────────────────────────────────────────────
@dataclass
class Chunk:
    """
    A single piece of text extracted from a source document.

    Attributes
    ----------
    chunk_id    : Unique ID (derived from doc_id + chunk index)
    doc_id      : Parent document UUID
    source      : Human-readable source label (file name + page)
    file_name   : Original file name
    page_number : Page / section number in the source document (or None)
    text        : The chunk text
    token_count : Token count (cl100k_base)
    char_start  : Character start offset within the page text
    char_end    : Character end offset within the page text
    metadata    : Arbitrary extra fields
    """

    chunk_id: str
    doc_id: str
    source: str
    file_name: str
    page_number: int | None
    text: str
    token_count: int
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            **self.metadata,
        }


# ─── Sentence splitter ───────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLTK punkt tokenizer.
    Falls back to a simple regex splitter if NLTK is unavailable.
    """
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Regex fallback
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


# ─── Token Chunker ───────────────────────────────────────────
class TokenChunker:
    """
    Split document pages into token-aware chunks with overlap.

    Algorithm
    ---------
    1. Split each page into sentences.
    2. Greedily accumulate sentences until the chunk would exceed
       ``chunk_size`` tokens.
    3. When the limit is reached, emit the chunk and backtrack by
       ``overlap`` tokens to create continuity.

    Parameters
    ----------
    chunk_size : int
        Maximum tokens per chunk.
    chunk_overlap : int
        Number of overlapping tokens from the previous chunk.
    min_chunk_length : int
        Minimum character length; shorter chunks are discarded.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self._enc = get_tokeniser()

    # ── public ───────────────────────────────────────────────
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Chunk an entire document, page by page.

        Returns a flat list of ``Chunk`` objects in reading order.
        """
        all_chunks: list[Chunk] = []
        chunk_idx = 0

        for page in doc.pages:
            page_chunks = self._chunk_page(
                page=page,
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                start_chunk_idx=chunk_idx,
            )
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        log.debug(
            f"Chunked document",
            file=doc.file_name,
            pages=doc.num_pages,
            chunks=len(all_chunks),
        )
        return all_chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        file_name: str,
        page_number: int | None = None,
    ) -> list[Chunk]:
        """Chunk a raw text string (convenience wrapper)."""
        page = PageContent(text=text, page_number=page_number)
        return self._chunk_page(page, doc_id, file_name, start_chunk_idx=0)

    # ── private ──────────────────────────────────────────────
    def _chunk_page(
        self,
        page: PageContent,
        doc_id: str,
        file_name: str,
        start_chunk_idx: int,
    ) -> list[Chunk]:
        sentences = _split_sentences(page.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens: int = 0
        chunk_local_idx = 0

        def _emit_chunk(sents: list[str]) -> Chunk | None:
            nonlocal chunk_local_idx
            text = " ".join(sents)
            if len(text) < self.min_chunk_length:
                return None
            tok_count = count_tokens(text)
            page_label = f"page {page.page_number}" if page.page_number else "section"
            source_label = f"{file_name} — {page_label}"
            c = Chunk(
                chunk_id=chunk_id(doc_id, start_chunk_idx + chunk_local_idx),
                doc_id=doc_id,
                source=source_label,
                file_name=file_name,
                page_number=page.page_number,
                text=text,
                token_count=tok_count,
                metadata=page.metadata,
            )
            chunk_local_idx += 1
            return c

        for sentence in sentences:
            sent_tokens = count_tokens(sentence)

            # If a single sentence exceeds chunk_size, split it by words
            if sent_tokens > self.chunk_size:
                # Flush current accumulation first
                if current_sentences:
                    ch = _emit_chunk(current_sentences)
                    if ch:
                        chunks.append(ch)
                    current_sentences = []
                    current_tokens = 0

                # Split the giant sentence into sub-chunks
                sub_chunks = self._split_long_sentence(sentence)
                for sub in sub_chunks:
                    ch = _emit_chunk([sub])
                    if ch:
                        chunks.append(ch)
                continue

            if current_tokens + sent_tokens > self.chunk_size:
                # Emit current chunk
                ch = _emit_chunk(current_sentences)
                if ch:
                    chunks.append(ch)

                # Carry over overlap sentences
                overlap_sents: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tok = count_tokens(s)
                    if overlap_tokens + s_tok <= self.chunk_overlap:
                        overlap_sents.insert(0, s)
                        overlap_tokens += s_tok
                    else:
                        break

                current_sentences = overlap_sents
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush remaining sentences
        if current_sentences:
            ch = _emit_chunk(current_sentences)
            if ch:
                chunks.append(ch)

        return chunks

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Split a sentence that exceeds chunk_size by word boundaries."""
        words = sentence.split()
        parts: list[str] = []
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            wt = count_tokens(word)
            if current_tokens + wt > self.chunk_size:
                if current_words:
                    parts.append(" ".join(current_words))
                current_words = [word]
                current_tokens = wt
            else:
                current_words.append(word)
                current_tokens += wt

        if current_words:
            parts.append(" ".join(current_words))
        return parts


# ─── Chunking pipeline ───────────────────────────────────────
def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_length: int = 50,
) -> list[Chunk]:
    """
    Chunk a list of documents and return all chunks in order.

    Parameters
    ----------
    docs            : Documents produced by DocumentLoader
    chunk_size      : Max tokens per chunk
    chunk_overlap   : Overlap tokens between adjacent chunks
    min_chunk_length: Drop chunks shorter than this many characters

    Returns
    -------
    list[Chunk]
        All chunks from all documents, ordered by document then page.
    """
    chunker = TokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_length=min_chunk_length,
    )
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    log.info(
        f"Chunking complete",
        documents=len(docs),
        total_chunks=len(all_chunks),
    )
    return all_chunks
