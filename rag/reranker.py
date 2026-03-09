"""
rag/reranker.py
─────────────────────────────────────────────────────────────
Cross-encoder re-ranking layer.

After the initial retriever returns a candidate set, the re-ranker
scores each (query, chunk) pair jointly using a cross-encoder model
(``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default).

Cross-encoders are much more accurate than bi-encoders for relevance
judgment — they look at query and passage together — but are too slow
to run over the entire corpus (hence retrieval→re-rank pipeline).

Features
--------
* Lazy model loading (only downloads on first use)
* Batch inference for efficiency
* Fallback to passthrough (no-op) when re-ranking is disabled
* Scores are min-max normalised to [0, 1]
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from rag.vector_store import SearchResult
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Model loader (singleton) ────────────────────────────────
@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    log.info(f"Loading cross-encoder: {model_name}")
    t0 = time.perf_counter()
    model = CrossEncoder(model_name, max_length=512)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(f"Cross-encoder loaded in {elapsed:.0f} ms")
    return model


# ─── Re-ranker ───────────────────────────────────────────────
class CrossEncoderReranker:
    """
    Re-rank a list of SearchResults using a cross-encoder model.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model identifier.
    top_k      : Number of results to return after re-ranking.
    batch_size : Inference batch size (reduce if OOM).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Score and re-rank *results* for *query*.

        Parameters
        ----------
        query   : User question.
        results : Candidate results from the retriever.

        Returns
        -------
        list[SearchResult]
            Re-ranked results, truncated to *top_k*, with updated scores
            and ranks.
        """
        if not results:
            return []

        t0 = time.perf_counter()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, r.chunk.text) for r in results]

        # Batch inference
        raw_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            scores = self.model.predict(batch, convert_to_numpy=True)
            raw_scores.extend(scores.tolist())

        # Min-max normalise scores to [0, 1]
        arr = np.array(raw_scores, dtype=np.float32)
        min_s, max_s = arr.min(), arr.max()
        if max_s > min_s:
            normalised = ((arr - min_s) / (max_s - min_s)).tolist()
        else:
            normalised = [1.0] * len(arr)

        # Sort by normalised score descending
        ranked_pairs = sorted(
            zip(results, normalised, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked: list[SearchResult] = []
        for rank, (result, score) in enumerate(
            ranked_pairs[: self.top_k], start=1
        ):
            result.score = round(score, 6)
            result.rank = rank
            reranked.append(result)

        elapsed = (time.perf_counter() - t0) * 1000
        log.debug(
            f"Re-ranked {len(results)} → {len(reranked)} results "
            f"in {elapsed:.0f} ms"
        )
        return reranked


# ─── No-op fallback ──────────────────────────────────────────
class PassthroughReranker:
    """
    Identity re-ranker — returns results unchanged.

    Used when re-ranking is disabled in config, allowing the pipeline
    to remain unchanged regardless of the re-ranking flag.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        return results[: self.top_k]


# ─── Factory ─────────────────────────────────────────────────
def build_reranker(
    enabled: bool = True,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> CrossEncoderReranker | PassthroughReranker:
    """
    Return the appropriate re-ranker based on the *enabled* flag.

    Parameters
    ----------
    enabled    : If False, returns a PassthroughReranker.
    model_name : Cross-encoder model to use.
    top_k      : Maximum results after re-ranking.
    """
    if enabled:
        log.info(f"Cross-encoder re-ranking enabled: {model_name}")
        return CrossEncoderReranker(model_name=model_name, top_k=top_k)
    else:
        log.info("Re-ranking disabled — using passthrough")
        return PassthroughReranker(top_k=top_k)
