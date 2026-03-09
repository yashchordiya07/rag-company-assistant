"""
rag/vector_store.py
─────────────────────────────────────────────────────────────
FAISS-backed vector store with persistent storage.

Responsibilities
----------------
* Index chunk embeddings for fast approximate nearest-neighbour search
* Store and retrieve chunk metadata (JSON-based side-car file)
* Persist the FAISS index and metadata to disk; reload on startup
* Provide similarity_search with optional score filtering
* Deduplicate documents by SHA-256 hash

Index type: ``IndexFlatIP`` (inner product ≡ cosine similarity for
L2-normalised vectors).  For production at scale, swap to
``IndexIVFFlat`` or ``IndexHNSW`` without changing the interface.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from rag.chunking import Chunk
from rag.embeddings import EmbeddingModel
from utils.logger import get_logger

log = get_logger(__name__)

_META_SUFFIX = "_metadata.json"
_INDEX_SUFFIX = ".faiss"
_HASHES_SUFFIX = "_hashes.json"


# ─── Result type ─────────────────────────────────────────────
class SearchResult:
    """A single result returned by the vector store."""

    __slots__ = ("chunk", "score", "rank")

    def __init__(self, chunk: Chunk, score: float, rank: int) -> None:
        self.chunk = chunk
        self.score = score      # cosine similarity ∈ [0, 1]
        self.rank = rank        # 1-based

    def to_dict(self) -> dict[str, Any]:
        d = self.chunk.to_dict()
        d["score"] = round(self.score, 6)
        d["rank"] = self.rank
        return d

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"source='{self.chunk.source}')"
        )


# ─── Vector Store ────────────────────────────────────────────
class FAISSVectorStore:
    """
    A FAISS vector store that stores chunk embeddings and metadata.

    Parameters
    ----------
    store_path : Path
        Directory where index files are persisted.
    index_name : str
        File prefix for the FAISS index and metadata files.
    embedding_model : EmbeddingModel
        Shared embedding model used to embed queries at search time.
    """

    def __init__(
        self,
        store_path: Path,
        index_name: str,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.store_path = Path(store_path)
        self.index_name = index_name
        self.embedder = embedding_model

        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []             # parallel to index rows
        self._doc_hashes: set[str] = set()         # for deduplication

        self.store_path.mkdir(parents=True, exist_ok=True)

    # ── Paths ─────────────────────────────────────────────────
    @property
    def _index_path(self) -> Path:
        return self.store_path / (self.index_name + _INDEX_SUFFIX)

    @property
    def _meta_path(self) -> Path:
        return self.store_path / (self.index_name + _META_SUFFIX)

    @property
    def _hashes_path(self) -> Path:
        return self.store_path / (self.index_name + _HASHES_SUFFIX)

    # ── Initialisation ────────────────────────────────────────
    def _build_index(self) -> faiss.Index:
        """Create a new flat inner-product FAISS index."""
        dim = self.embedder.embedding_dim
        index = faiss.IndexFlatIP(dim)
        # Wrap in an IDMap so we can assign sequential integer IDs
        return faiss.IndexIDMap(index)

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    # ── Add documents ─────────────────────────────────────────
    def add_chunks(
        self,
        chunk_embedding_pairs: list[tuple[Chunk, NDArray[np.float32]]],
        doc_hash: str | None = None,
    ) -> int:
        """
        Add a batch of (Chunk, vector) pairs to the index.

        Parameters
        ----------
        chunk_embedding_pairs : list of (Chunk, ndarray) tuples
        doc_hash              : Optional SHA-256 of the source document.
                                If provided and already indexed, the batch
                                is skipped (deduplication).

        Returns
        -------
        int
            Number of chunks actually added.
        """
        if doc_hash and doc_hash in self._doc_hashes:
            log.info(f"Document already indexed (hash={doc_hash[:8]}…), skipping.")
            return 0

        if not chunk_embedding_pairs:
            return 0

        chunks, vectors = zip(*chunk_embedding_pairs, strict=True)
        matrix = np.vstack(vectors).astype(np.float32)

        # IDs are sequential integers starting after existing entries
        start_id = len(self._chunks)
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)

        self.index.add_with_ids(matrix, ids)  # type: ignore[attr-defined]
        self._chunks.extend(chunks)

        if doc_hash:
            self._doc_hashes.add(doc_hash)

        log.info(
            f"Added {len(chunks)} chunks to index "
            f"(total={len(self._chunks)})"
        )
        return len(chunks)

    def add_documents_from_embedder(
        self,
        chunks: list[Chunk],
        doc_hash: str | None = None,
        show_progress: bool = True,
    ) -> int:
        """Embed *chunks* using the embedder then add them to the store."""
        pairs = self.embedder.embed_chunks(chunks, show_progress=show_progress)
        return self.add_chunks(pairs, doc_hash=doc_hash)

    # ── Search ────────────────────────────────────────────────
    def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find the *top_k* most similar chunks to *query*.

        Parameters
        ----------
        query           : User question / search string.
        top_k           : Maximum number of results.
        score_threshold : Discard results with score < threshold.

        Returns
        -------
        list[SearchResult]
            Ranked results (highest score first).
        """
        if len(self._chunks) == 0:
            log.warning("Vector store is empty — no results returned.")
            return []

        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        actual_k = min(top_k, len(self._chunks))

        scores, ids = self.index.search(query_vec, actual_k)  # type: ignore[attr-defined]
        scores = scores[0].tolist()
        ids = ids[0].tolist()

        results: list[SearchResult] = []
        rank = 1
        for score, idx in zip(scores, ids, strict=True):
            if idx < 0 or idx >= len(self._chunks):
                continue
            if score < score_threshold:
                continue
            results.append(SearchResult(chunk=self._chunks[idx], score=score, rank=rank))
            rank += 1

        return results

    def similarity_search_by_vector(
        self,
        query_vec: NDArray[np.float32],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Like ``similarity_search`` but accepts a pre-computed query vector."""
        if len(self._chunks) == 0:
            return []

        q = query_vec.reshape(1, -1).astype(np.float32)
        actual_k = min(top_k, len(self._chunks))
        scores, ids = self.index.search(q, actual_k)  # type: ignore[attr-defined]
        scores = scores[0].tolist()
        ids = ids[0].tolist()

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores, ids, strict=True), start=1):
            if idx < 0 or idx >= len(self._chunks):
                continue
            if score < score_threshold:
                continue
            results.append(SearchResult(chunk=self._chunks[idx], score=score, rank=rank))
        return results

    # ── Persistence ───────────────────────────────────────────
    def save(self) -> None:
        """Persist index, chunk metadata, and document hashes to disk."""
        if self._index is None:
            log.warning("Nothing to save — index has not been built yet.")
            return

        # Save FAISS index
        faiss.write_index(self._index, str(self._index_path))

        # Save chunk metadata (serialise to JSON)
        meta = [c.to_dict() for c in self._chunks]
        self._meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
        )

        # Save document hashes
        self._hashes_path.write_text(
            json.dumps(list(self._doc_hashes), ensure_ascii=False), encoding="utf-8"
        )

        log.info(
            f"Vector store saved",
            index=str(self._index_path),
            chunks=len(self._chunks),
        )

    def load(self) -> bool:
        """
        Load a previously persisted index from disk.

        Returns
        -------
        bool
            True if loaded successfully, False if no index found on disk.
        """
        if not self._index_path.exists() or not self._meta_path.exists():
            log.info("No persisted vector store found — starting fresh.")
            return False

        self._index = faiss.read_index(str(self._index_path))

        raw_meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._chunks = [_dict_to_chunk(m) for m in raw_meta]

        if self._hashes_path.exists():
            self._doc_hashes = set(
                json.loads(self._hashes_path.read_text(encoding="utf-8"))
            )

        log.info(
            f"Vector store loaded",
            chunks=len(self._chunks),
            index_type=type(self._index).__name__,
        )
        return True

    def clear(self) -> None:
        """Delete all data from the store (in-memory and on disk)."""
        self._index = None
        self._chunks = []
        self._doc_hashes = set()
        for path in [self._index_path, self._meta_path, self._hashes_path]:
            if path.exists():
                path.unlink()
        log.warning("Vector store cleared.")

    def backup(self, backup_dir: str | Path) -> None:
        """Copy the current store files to *backup_dir*."""
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in [self._index_path, self._meta_path, self._hashes_path]:
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
        log.info(f"Vector store backed up to {backup_dir}")

    # ── Stats ─────────────────────────────────────────────────
    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_documents(self) -> int:
        return len({c.doc_id for c in self._chunks})

    def stats(self) -> dict[str, Any]:
        """Return a summary of the current store state."""
        return {
            "num_chunks": self.num_chunks,
            "num_documents": self.num_documents,
            "num_indexed_doc_hashes": len(self._doc_hashes),
            "index_type": type(self._index).__name__ if self._index else "None",
            "embedding_dim": self.embedder.embedding_dim,
            "store_path": str(self.store_path),
        }


# ─── Deserialisation helper ──────────────────────────────────
def _dict_to_chunk(d: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from its serialised dict representation."""
    return Chunk(
        chunk_id=d.get("chunk_id", ""),
        doc_id=d.get("doc_id", ""),
        source=d.get("source", ""),
        file_name=d.get("file_name", ""),
        page_number=d.get("page_number"),
        text=d.get("text", ""),
        token_count=d.get("token_count", 0),
        char_start=d.get("char_start", 0),
        char_end=d.get("char_end", 0),
        metadata={
            k: v
            for k, v in d.items()
            if k
            not in {
                "chunk_id", "doc_id", "source", "file_name",
                "page_number", "text", "token_count", "char_start", "char_end",
            }
        },
    )
