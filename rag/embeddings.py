"""
rag/embeddings.py
─────────────────────────────────────────────────────────────
Embedding layer for the RAG pipeline.

Uses SentenceTransformers (``all-MiniLM-L6-v2`` by default) to
produce dense vector representations of text chunks and queries.

Features
--------
* Model is loaded once and reused (singleton pattern)
* Batch encoding with a progress bar for large corpora
* L2-normalisation so cosine similarity ≡ dot product
* Async-compatible (embeddings run in a thread-pool executor)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from rag.chunking import Chunk
from utils.logger import get_logger, log_performance

log = get_logger(__name__)

# Thread pool for async embedding calls
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")


# ─── Model loader (singleton) ────────────────────────────────
@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> SentenceTransformer:
    """Load and cache a SentenceTransformer model."""
    log.info(f"Loading embedding model: {model_name} on {device}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(f"Embedding model loaded in {elapsed:.0f} ms")
    return model


# ─── Embedder ────────────────────────────────────────────────
class EmbeddingModel:
    """
    Wrapper around a SentenceTransformer that provides:
    - ``embed_texts``  : encode a list of strings → float32 matrix
    - ``embed_query``  : encode a single query string → 1-D float32 array
    - ``embed_chunks`` : encode Chunk objects, return (Chunk, vector) pairs
    - Async variants of all of the above

    Parameters
    ----------
    model_name : str
        SentenceTransformer model identifier.
    device     : str
        "cpu" or "cuda".
    batch_size : int
        Number of texts per encoding batch.
    normalize  : bool
        L2-normalise vectors (required for cosine similarity via FAISS
        IndexFlatIP / inner product).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding vectors produced by this model."""
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    # ── Synchronous API ──────────────────────────────────────
    @log_performance
    def embed_texts(self, texts: Sequence[str], show_progress: bool = False) -> NDArray[np.float32]:
        """
        Encode a list of texts into a 2-D float32 matrix of shape
        (len(texts), embedding_dim).

        Parameters
        ----------
        texts         : Sequence of strings to embed.
        show_progress : Show a tqdm progress bar (useful for large batches).

        Returns
        -------
        NDArray[np.float32]
            Normalised embedding matrix.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        t0 = time.perf_counter()
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug(
            f"Embedded {len(texts)} texts in {elapsed:.0f} ms "
            f"({elapsed / max(len(texts), 1):.1f} ms/text)"
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """
        Encode a single query string into a 1-D float32 vector.

        Returns
        -------
        NDArray[np.float32]
            Shape: (embedding_dim,)
        """
        return self.embed_texts([query])[0]

    def embed_chunks(
        self, chunks: list[Chunk], show_progress: bool = True
    ) -> list[tuple[Chunk, NDArray[np.float32]]]:
        """
        Embed a list of Chunk objects.

        Returns
        -------
        list[tuple[Chunk, NDArray]]
            Pairs of (chunk, vector).
        """
        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts, show_progress=show_progress)
        return list(zip(chunks, vectors, strict=True))

    # ── Async API ────────────────────────────────────────────
    async def aembed_texts(
        self, texts: Sequence[str], show_progress: bool = False
    ) -> NDArray[np.float32]:
        """Async version of embed_texts (runs in a thread-pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _EXECUTOR, lambda: self.embed_texts(texts, show_progress)
        )

    async def aembed_query(self, query: str) -> NDArray[np.float32]:
        """Async version of embed_query."""
        return (await self.aembed_texts([query]))[0]

    # ── Utility ──────────────────────────────────────────────
    @staticmethod
    def cosine_similarity(
        vec_a: NDArray[np.float32],
        vec_b: NDArray[np.float32],
    ) -> float:
        """
        Compute cosine similarity between two normalised vectors.
        (For L2-normalised vectors this equals the dot product.)
        """
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def batch_cosine_similarity(
        query_vec: NDArray[np.float32],
        corpus_vecs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Compute cosine similarities between a single query vector and
        a matrix of corpus vectors.

        Parameters
        ----------
        query_vec    : Shape (dim,)
        corpus_vecs  : Shape (n, dim)

        Returns
        -------
        NDArray[np.float32]
            Shape (n,) — similarity score for each corpus vector.
        """
        return corpus_vecs @ query_vec


# ─── Convenience factory ─────────────────────────────────────
_DEFAULT_EMBEDDER: EmbeddingModel | None = None


def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> EmbeddingModel:
    """Return a shared EmbeddingModel instance (lazy singleton)."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model_name != model_name:
        _DEFAULT_EMBEDDER = EmbeddingModel(model_name=model_name, device=device)
    return _DEFAULT_EMBEDDER
