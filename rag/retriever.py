"""
rag/retriever.py
─────────────────────────────────────────────────────────────
Retriever layer — combines semantic (FAISS) and lexical (BM25) search
using Reciprocal Rank Fusion (RRF) when hybrid mode is enabled.

Components
----------
* ``SemanticRetriever``  — pure FAISS cosine similarity search
* ``BM25Retriever``      — keyword matching (rank-bm25)
* ``HybridRetriever``    — fuses both signals with configurable weights
* ``QueryRewriter``      — uses a lightweight template to rewrite queries
                           with conversation context before retrieval
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.vector_store import FAISSVectorStore, SearchResult
from utils.logger import get_logger, log_retrieval_event

log = get_logger(__name__)


# ─── Semantic Retriever ───────────────────────────────────────
class SemanticRetriever:
    """Pure vector-similarity retriever backed by FAISS."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[SearchResult]:
        return self.vector_store.similarity_search(
            query=query,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )


# ─── BM25 Retriever ──────────────────────────────────────────
class BM25Retriever:
    """
    Keyword-based retriever using BM25Okapi.

    Maintains a BM25 index that mirrors the FAISS vector store.
    It must be rebuilt (or updated) whenever the vector store changes.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks."""
        self._chunks = chunks
        tokenised = [_simple_tokenise(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        log.info(f"BM25 index built with {len(chunks)} chunks")

    def retrieve(self, query: str) -> list[SearchResult]:
        if self._bm25 is None or not self._chunks:
            log.warning("BM25 index not built — returning empty results")
            return []

        query_tokens = _simple_tokenise(query)
        raw_scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices by score (descending)
        top_indices = np.argsort(raw_scores)[::-1][: self.top_k]
        max_score = float(raw_scores.max()) or 1.0  # avoid div-by-zero

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(raw_scores[idx]) / max_score  # normalise to [0,1]
            if score <= 0:
                break
            results.append(
                SearchResult(chunk=self._chunks[idx], score=score, rank=rank)
            )
        return results


# ─── Hybrid Retriever (RRF fusion) ───────────────────────────
class HybridRetriever:
    """
    Combines semantic (FAISS) and BM25 results using Reciprocal Rank
    Fusion (RRF).

    RRF score = Σ 1 / (k + rank_i)

    where k=60 is a constant that dampens the influence of top ranks.

    Final score is a weighted average of normalised semantic and BM25
    scores, re-ranked by RRF for robustness.

    Parameters
    ----------
    vector_store    : FAISSVectorStore — semantic search backend.
    bm25_retriever  : BM25Retriever — keyword search backend.
    top_k           : Maximum candidates to retrieve per modality.
    top_k_final     : Final number of results returned after fusion.
    semantic_weight : Weight for semantic scores (0–1).
    bm25_weight     : Weight for BM25 scores (0–1).
    score_threshold : Minimum blended score to include a result.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        top_k: int = 10,
        top_k_final: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        score_threshold: float = 0.0,
    ) -> None:
        self.semantic = SemanticRetriever(vector_store, top_k, score_threshold)
        self.bm25 = bm25_retriever
        self.top_k = top_k
        self.top_k_final = top_k_final
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[SearchResult]:
        """
        Retrieve using both modalities and fuse the results.

        Returns
        -------
        list[SearchResult]
            Fused and re-ranked results, capped at *top_k_final*.
        """
        sem_results = self.semantic.retrieve(query)
        bm25_results = self.bm25.retrieve(query)

        fused = _rrf_fusion(
            sem_results,
            bm25_results,
            sem_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
        )

        # Re-assign ranks after fusion
        for rank, r in enumerate(fused, start=1):
            r.rank = rank

        return fused[: self.top_k_final]


# ─── Query Rewriter ──────────────────────────────────────────
class QueryRewriter:
    """
    Rewrites a user query using conversation history to make it
    self-contained before retrieval.

    Uses a simple rule-based approach for speed; can be swapped to
    an LLM-based approach in production.

    Heuristics
    ----------
    * If query contains pronouns (he/she/it/they/this/that) AND
      there is conversation history, prepend context from the last
      assistant message.
    * If query starts with "and", "also", "but" etc., append the
      subject of the last user question.
    """

    PRONOUN_PATTERN = re.compile(
        r"\b(he|she|it|they|this|that|these|those|him|her|them|its)\b",
        re.IGNORECASE,
    )

    CONTINUATION_WORDS = {"and", "also", "but", "so", "furthermore", "additionally"}

    def rewrite(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        """
        Rewrite *query* using conversation *history*.

        Parameters
        ----------
        query   : Current user question.
        history : List of {"role": ..., "content": ...} dicts.

        Returns
        -------
        str
            Possibly rewritten query, or the original query unchanged.
        """
        if not history:
            return query

        first_word = query.split()[0].lower() if query.split() else ""
        has_pronoun = bool(self.PRONOUN_PATTERN.search(query))
        is_continuation = first_word in self.CONTINUATION_WORDS

        if not (has_pronoun or is_continuation):
            return query

        # Find last user message as context anchor
        last_user = next(
            (h["content"] for h in reversed(history) if h["role"] == "user"),
            "",
        )
        if not last_user or last_user == query:
            return query

        # Extract the noun phrase from the last user question (simple heuristic)
        subject = _extract_subject(last_user)
        if subject:
            rewritten = f"{subject} — {query}"
            log.debug(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten

        return query


# ─── Retriever Factory ───────────────────────────────────────
class RetrieverFactory:
    """Creates the appropriate retriever based on config flags."""

    @staticmethod
    def build(
        vector_store: FAISSVectorStore,
        chunks: list[Chunk],
        *,
        hybrid: bool = True,
        top_k: int = 10,
        top_k_final: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        score_threshold: float = 0.0,
    ) -> HybridRetriever | SemanticRetriever:
        """
        Build and return the retriever.

        Parameters
        ----------
        vector_store    : Populated FAISSVectorStore.
        chunks          : All chunks (required to build BM25 index).
        hybrid          : If True, use HybridRetriever; else SemanticRetriever.
        ...             : Forwarded to the retriever constructor.
        """
        if hybrid and chunks:
            bm25 = BM25Retriever(top_k=top_k)
            bm25.build_index(chunks)
            retriever = HybridRetriever(
                vector_store=vector_store,
                bm25_retriever=bm25,
                top_k=top_k,
                top_k_final=top_k_final,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                score_threshold=score_threshold,
            )
            log.info("Hybrid retriever (semantic + BM25) initialised")
        else:
            retriever = SemanticRetriever(
                vector_store=vector_store,
                top_k=top_k_final,
                score_threshold=score_threshold,
            )
            log.info("Semantic-only retriever initialised")
        return retriever


# ─── Private helpers ─────────────────────────────────────────
def _simple_tokenise(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _rrf_fusion(
    sem_results: list[SearchResult],
    bm25_results: list[SearchResult],
    *,
    sem_weight: float = 0.7,
    bm25_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Deduplicates by chunk_id; assigns a blended score that combines
    RRF and the original cosine / BM25 scores.
    """
    # Map chunk_id → SearchResult (keep highest-scoring duplicate)
    chunk_map: dict[str, SearchResult] = {}
    sem_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for r in sem_results:
        cid = r.chunk.chunk_id
        sem_rank[cid] = r.rank
        chunk_map[cid] = r

    for r in bm25_results:
        cid = r.chunk.chunk_id
        bm25_rank[cid] = r.rank
        if cid not in chunk_map or r.score > chunk_map[cid].score:
            chunk_map[cid] = r

    # Compute blended scores
    blended: dict[str, float] = defaultdict(float)
    sem_scores: dict[str, float] = {r.chunk.chunk_id: r.score for r in sem_results}
    bm25_scores: dict[str, float] = {r.chunk.chunk_id: r.score for r in bm25_results}

    all_cids = set(sem_rank) | set(bm25_rank)
    for cid in all_cids:
        rrf_sem = 1 / (rrf_k + sem_rank.get(cid, 1000))
        rrf_bm25 = 1 / (rrf_k + bm25_rank.get(cid, 1000))
        # Weighted RRF
        blended[cid] = sem_weight * rrf_sem + bm25_weight * rrf_bm25
        # Bonus: if both modalities agree, boost score
        if cid in sem_rank and cid in bm25_rank:
            blended[cid] *= 1.1

    # Sort by blended score descending
    ranked_cids = sorted(blended, key=blended.__getitem__, reverse=True)

    results: list[SearchResult] = []
    for rank, cid in enumerate(ranked_cids, start=1):
        r = chunk_map[cid]
        # Normalise score to [0, 1] using max blended score
        max_blended = max(blended.values()) or 1.0
        r.score = blended[cid] / max_blended
        r.rank = rank
        results.append(r)

    return results


def _extract_subject(text: str) -> str:
    """Very simple noun-phrase extractor for query rewriting."""
    # Try to get the main noun phrase from the question
    # Pattern: "What is/are the <NP>?" → return <NP>
    match = re.search(
        r"(?:what is|what are|tell me about|explain|describe)\s+(?:the\s+)?([^?.,]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: return first 4–6 words
    words = text.split()
    return " ".join(words[:5]) if len(words) > 5 else ""
