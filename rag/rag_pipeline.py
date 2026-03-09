"""
rag/rag_pipeline.py
─────────────────────────────────────────────────────────────
The central RAG pipeline — orchestrates every component.

Pipeline flow
─────────────
1.  User question arrives.
2.  QueryRewriter rewrites the query using conversation history.
3.  HybridRetriever (FAISS + BM25) retrieves top-k candidates.
4.  CrossEncoderReranker re-ranks candidates.
5.  Context block is built from the final results.
6.  Prompt is assembled (system + context + history + question).
7.  LLM generates an answer.
8.  Citations are extracted and attached to the response.
9.  The Q/A turn is stored in conversation memory.

Caching
───────
Query results are cached by MD5(query + top_k) using an in-process
LRU dict (or Redis when ENABLE_CACHE=true).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from rag.chunking import Chunk, chunk_documents
from rag.document_loader import Document, DocumentLoader
from rag.embeddings import EmbeddingModel, get_embedding_model
from rag.reranker import build_reranker
from rag.retriever import HybridRetriever, QueryRewriter, RetrieverFactory, SemanticRetriever
from rag.vector_store import FAISSVectorStore, SearchResult
from models.llm_model import (
    LLMBase,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    build_llm,
)
from utils.helpers import build_context_block, extract_citations, text_hash
from utils.logger import get_logger, log_retrieval_event

log = get_logger(__name__)


# ─── Response model ──────────────────────────────────────────
@dataclass
class RAGResponse:
    """Structured output of a single RAG query."""

    question: str
    answer: str
    rewritten_question: str
    retrieved_chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    latency_ms: float
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "answer": self.answer,
            "retrieved_chunks": self.retrieved_chunks,
            "citations": self.citations,
            "latency_ms": round(self.latency_ms, 1),
            "from_cache": self.from_cache,
            **self.metadata,
        }


# ─── In-process cache (LRU dict) ─────────────────────────────
class _InMemoryCache:
    """Simple LRU cache backed by a dict + deque."""

    def __init__(self, max_size: int = 256, ttl: int = 3600) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._order: deque[str] = deque()
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self.ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._order.popleft()
            self._cache.pop(oldest, None)
        self._cache[key] = (value, time.time())
        self._order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


# ─── Conversation Memory ─────────────────────────────────────
class ConversationMemory:
    """
    Stores the last N turns of a conversation (user + assistant).

    Each turn is a dict: {"role": "user"|"assistant", "content": str}
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._history: deque[dict[str, str]] = deque(maxlen=max_turns * 2)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": assistant_message})

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def format_for_display(self) -> list[dict[str, str]]:
        """Return history in user-friendly format."""
        return self.history


# ─── RAG Pipeline ────────────────────────────────────────────
class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage
    -----
    pipeline = RAGPipeline.from_config(settings)
    pipeline.ingest_document("path/to/policy.pdf")
    response = pipeline.query("What is the vacation policy?")
    print(response.answer)
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        vector_store: FAISSVectorStore,
        llm: LLMBase,
        retriever: HybridRetriever | SemanticRetriever | None = None,
        reranker: Any = None,
        query_rewriter: QueryRewriter | None = None,
        memory: ConversationMemory | None = None,
        cache: _InMemoryCache | None = None,
        enable_cache: bool = False,
        enable_reranking: bool = True,
        enable_hybrid: bool = True,
        enable_query_rewriting: bool = True,
        top_k_retrieval: int = 10,
        top_k_final: int = 5,
        similarity_threshold: float = 0.3,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.memory = memory or ConversationMemory()
        self._cache = cache
        self.enable_cache = enable_cache
        self.enable_reranking = enable_reranking
        self.enable_query_rewriting = enable_query_rewriting
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.similarity_threshold = similarity_threshold

        # Retriever (built lazily after first document is ingested)
        self._retriever = retriever
        self._retriever_needs_rebuild = True
        self._all_chunks: list[Chunk] = []

        # Config params for retriever rebuilding
        self._enable_hybrid = enable_hybrid
        self._bm25_weight = bm25_weight
        self._semantic_weight = semantic_weight

        if enable_cache and self._cache is None:
            self._cache = _InMemoryCache()

        log.info("RAG pipeline initialised")

    # ── Document ingestion ────────────────────────────────────
    def ingest_document(
        self,
        path: str | Path | None = None,
        data: bytes | None = None,
        file_name: str = "upload",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> dict[str, Any]:
        """
        Load, chunk, embed, and index a document.

        Parameters
        ----------
        path        : File path on disk (optional if *data* provided).
        data        : Raw file bytes (required if *path* not provided).
        file_name   : Used for metadata when *data* provided.
        chunk_size  : Tokens per chunk.
        chunk_overlap : Overlap tokens between chunks.

        Returns
        -------
        dict with ingestion stats.
        """
        t0 = time.perf_counter()
        loader = DocumentLoader()

        if path is not None:
            doc = loader.load(path, data=data)
        else:
            # Build a temporary Path from file_name for extension detection
            doc = loader.load(Path(file_name), data=data)

        from rag.chunking import TokenChunker
        chunker = TokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk_document(doc)

        added = self.vector_store.add_documents_from_embedder(
            chunks=chunks,
            doc_hash=doc.file_hash,
            show_progress=True,
        )
        self.vector_store.save()

        self._all_chunks.extend(chunks)
        self._retriever_needs_rebuild = True

        if self.enable_cache and self._cache:
            self._cache.clear()  # Invalidate cache on new data

        elapsed = (time.perf_counter() - t0) * 1000
        stats = {
            "doc_id": doc.doc_id,
            "file_name": doc.file_name,
            "pages": doc.num_pages,
            "chunks": len(chunks),
            "chunks_indexed": added,
            "char_count": doc.char_count,
            "latency_ms": round(elapsed, 1),
            "skipped": added == 0,
        }
        log.info("Document ingested", **stats)
        return stats

    def ingest_directory(self, directory: str | Path) -> list[dict[str, Any]]:
        """Ingest all supported documents from a directory."""
        loader = DocumentLoader()
        docs = loader.load_directory(directory)
        return [self.ingest_document(path=d.source_path) for d in docs]

    # ── Query ─────────────────────────────────────────────────
    def query(
        self,
        question: str,
        session_id: str = "default",
        stream: bool = False,
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Parameters
        ----------
        question   : User's natural language question.
        session_id : Conversation session identifier (for memory routing).
        stream     : If True, print tokens as they arrive (console demo).

        Returns
        -------
        RAGResponse
        """
        t0 = time.perf_counter()

        # ── 1. Query rewriting ───────────────────────────────
        rewritten = question
        if self.enable_query_rewriting:
            rewritten = self.query_rewriter.rewrite(
                question, self.memory.history
            )

        # ── 2. Cache lookup ──────────────────────────────────
        cache_key = _cache_key(rewritten, self.top_k_final)
        if self.enable_cache and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                log.info("Cache hit", key=cache_key[:8])
                cached["from_cache"] = True
                return RAGResponse(**cached)

        # ── 3. Retrieval ─────────────────────────────────────
        retriever = self._get_retriever()
        candidates: list[SearchResult] = retriever.retrieve(rewritten)

        # ── 4. Re-ranking ────────────────────────────────────
        if self.enable_reranking and self.reranker and candidates:
            results = self.reranker.rerank(rewritten, candidates)
        else:
            results = candidates[: self.top_k_final]

        # ── 5. Build context ─────────────────────────────────
        chunk_dicts = [r.to_dict() for r in results]
        context = build_context_block(
            [{"text": r.chunk.text, "source": r.chunk.source, "page": r.chunk.page_number}
             for r in results]
        )

        log_retrieval_event(
            query=rewritten,
            num_candidates=len(candidates),
            num_final=len(results),
            top_score=results[0].score if results else 0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

        # ── 6. Prompt assembly ───────────────────────────────
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=rewritten,
        )

        # ── 7. LLM generation ────────────────────────────────
        answer: str
        if stream:
            tokens = []
            for token in self.llm.stream(
                prompt=user_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                history=self.memory.history[-6:],  # last 3 turns
            ):
                tokens.append(token)
                print(token, end="", flush=True)
            print()
            answer = "".join(tokens)
        else:
            answer = self.llm.generate(
                prompt=user_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                history=self.memory.history[-6:],
            )

        # ── 8. Citations ─────────────────────────────────────
        citations = extract_citations(
            answer,
            [{"text": r.chunk.text, "source": r.chunk.source, "page": r.chunk.page_number}
             for r in results],
        )

        # ── 9. Memory update ─────────────────────────────────
        self.memory.add_turn(question, answer)

        elapsed = (time.perf_counter() - t0) * 1000
        response = RAGResponse(
            question=question,
            answer=answer,
            rewritten_question=rewritten,
            retrieved_chunks=chunk_dicts,
            citations=citations,
            latency_ms=elapsed,
            from_cache=False,
        )

        # Cache the result
        if self.enable_cache and self._cache:
            self._cache.set(cache_key, response.__dict__)

        return response

    async def aquery(self, question: str, session_id: str = "default") -> RAGResponse:
        """Async version of query (runs in thread executor)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question, session_id))

    # ── Retriever management ──────────────────────────────────
    def _get_retriever(self) -> HybridRetriever | SemanticRetriever:
        if self._retriever is None or self._retriever_needs_rebuild:
            self._retriever = RetrieverFactory.build(
                vector_store=self.vector_store,
                chunks=self._all_chunks,
                hybrid=self._enable_hybrid,
                top_k=self.top_k_retrieval,
                top_k_final=self.top_k_final,
                semantic_weight=self._semantic_weight,
                bm25_weight=self._bm25_weight,
                score_threshold=self.similarity_threshold,
            )
            self._retriever_needs_rebuild = False
        return self._retriever

    def rebuild_retriever(self) -> None:
        """Force rebuild of the retriever (e.g., after adding documents)."""
        self._retriever_needs_rebuild = True
        self._get_retriever()

    # ── Utility ───────────────────────────────────────────────
    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
        log.info("Conversation memory cleared")

    def stats(self) -> dict[str, Any]:
        return {
            "vector_store": self.vector_store.stats(),
            "total_chunks_in_memory": len(self._all_chunks),
            "llm_provider": type(self.llm).__name__,
            "reranking_enabled": self.enable_reranking,
            "hybrid_search_enabled": self._enable_hybrid,
            "cache_enabled": self.enable_cache,
        }

    # ── Factory ───────────────────────────────────────────────
    @classmethod
    def from_config(cls, settings: Any) -> "RAGPipeline":
        """
        Build a fully configured RAGPipeline from a Settings object.

        Parameters
        ----------
        settings : app.config.Settings instance
        """
        from utils.logger import setup_logging
        setup_logging(str(settings.log_level), settings.log_file)

        embedder = get_embedding_model(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        vector_store = FAISSVectorStore(
            store_path=settings.vector_store_path,
            index_name=settings.faiss_index_name,
            embedding_model=embedder,
        )

        # Try to load persisted index
        loaded = vector_store.load()
        all_chunks: list[Chunk] = []
        if loaded:
            # Reconstruct chunk list from vector store metadata
            all_chunks = vector_store._chunks[:]

        llm = build_llm(
            provider=settings.llm_provider,
            model=settings.openai_model if settings.llm_provider == "openai" else
                  settings.ollama_model if settings.llm_provider == "ollama" else
                  settings.huggingface_model,
            api_key=settings.openai_api_key,
            base_url=settings.ollama_base_url,
            api_token=settings.huggingface_api_token,
        )

        reranker = build_reranker(
            enabled=settings.enable_reranking,
            model_name=settings.reranker_model,
            top_k=settings.top_k_final,
        )

        cache: _InMemoryCache | None = None
        if settings.enable_cache:
            cache = _InMemoryCache(ttl=settings.cache_ttl_seconds)

        pipeline = cls(
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            reranker=reranker,
            enable_cache=settings.enable_cache,
            enable_reranking=settings.enable_reranking,
            enable_hybrid=settings.enable_hybrid_search,
            enable_query_rewriting=settings.enable_query_rewriting,
            top_k_retrieval=settings.top_k_retrieval,
            top_k_final=settings.top_k_final,
            similarity_threshold=settings.similarity_threshold,
            bm25_weight=settings.bm25_weight,
            semantic_weight=settings.semantic_weight,
        )
        pipeline._all_chunks = all_chunks
        return pipeline


# ─── Helpers ─────────────────────────────────────────────────
def _cache_key(query: str, top_k: int) -> str:
    payload = f"{query}||{top_k}"
    return hashlib.md5(payload.encode()).hexdigest()
