"""
ui/streamlit_app.py
─────────────────────────────────────────────────────────────
Streamlit-based chat interface for the Private Knowledge AI Assistant.

Features
--------
* Chat interface with streaming-style message display
* Document upload panel (PDF, DOCX, TXT, MD)
* Citation viewer — shows source chunks used to answer each question
* Session-based conversation memory (persisted in st.session_state)
* System stats panel in the sidebar
* Clear chat and clear index controls

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any

import httpx
import streamlit as st

# ─── Configuration ───────────────────────────────────────────
import os
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"
DEFAULT_TIMEOUT = 120  # seconds — long for first query (model load)

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Company Knowledge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background: #f0f4ff;
        border-left: 4px solid #4a90d9;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .chunk-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: #333;
    }
    .score-badge {
        background: #4a90d9;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .latency-text {
        color: #888;
        font-size: 0.8rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session state initialisation ────────────────────────────
def init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],          # list of {"role", "content", "metadata"}
        "session_id": f"sess_{int(time.time())}",
        "doc_count": 0,
        "last_retrieval": [],
        "last_citations": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ─── API helpers ─────────────────────────────────────────────
def api_health() -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{API_V1}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_stats() -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{API_V1}/stats", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_upload(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            f"{API_V1}/upload_document",
            files={"file": (file_name, BytesIO(file_bytes), _mime_type(file_name))},
        )
    response.raise_for_status()
    return response.json()


def api_query(question: str, session_id: str, top_k: int = 5) -> dict[str, Any]:
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            f"{API_V1}/query",
            json={
                "question": question,
                "session_id": session_id,
                "top_k": top_k,
            },
        )
    response.raise_for_status()
    return response.json()


def api_clear_memory() -> None:
    httpx.delete(f"{API_V1}/memory", timeout=10)


def api_clear_index() -> None:
    httpx.delete(f"{API_V1}/index", timeout=10)


def _mime_type(file_name: str) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower()
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "md": "text/markdown",
    }.get(ext, "application/octet-stream")


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Knowledge AI")
    st.markdown("*Private Company Assistant*")
    st.divider()

    # ── Connection status ─────────────────────────────────────
    st.markdown("### 🔌 Backend")
    health = api_health()
    if health:
        st.success("✅ API online")
    else:
        st.error("❌ API offline — start the backend:\n```\nuvicorn app.main:app\n```")

    # ── Stats ─────────────────────────────────────────────────
    if health:
        stats = api_stats()
        if stats:
            vs = stats.get("pipeline_stats", {}).get("vector_store", {})
            st.markdown("### 📊 Knowledge Base")
            col1, col2 = st.columns(2)
            col1.metric("Documents", vs.get("num_documents", 0))
            col2.metric("Chunks", vs.get("num_chunks", 0))

            cfg = stats.get("settings", {})
            with st.expander("⚙️ Config"):
                st.json(cfg)

    st.divider()

    # ── Document upload ───────────────────────────────────────
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Files are chunked and indexed into the vector store.",
    )

    if uploaded_files and st.button("📤 Ingest Documents", use_container_width=True):
        for uf in uploaded_files:
            with st.spinner(f"Ingesting {uf.name}…"):
                try:
                    result = api_upload(uf.read(), uf.name)
                    if result.get("skipped"):
                        st.info(f"⏭ {uf.name} — already indexed")
                    else:
                        st.success(
                            f"✅ {uf.name}\n"
                            f"Pages: {result['pages']} | "
                            f"Chunks: {result['chunks_indexed']}"
                        )
                        st.session_state.doc_count += 1
                except httpx.HTTPStatusError as exc:
                    st.error(f"❌ {uf.name}: {exc.response.json().get('detail', exc)}")
                except Exception as exc:
                    st.error(f"❌ {uf.name}: {exc}")

    st.divider()

    # ── Controls ──────────────────────────────────────────────
    st.markdown("### 🎛️ Controls")
    top_k = st.slider(
        "Retrieved chunks",
        min_value=1, max_value=10, value=5,
        help="More chunks → richer context but slower.",
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieval = []
        st.session_state.last_citations = []
        try:
            api_clear_memory()
        except Exception:
            pass
        st.rerun()

    with st.expander("⚠️ Admin"):
        if st.button("🔥 Wipe Index", use_container_width=True, type="primary"):
            try:
                api_clear_index()
                st.session_state.doc_count = 0
                st.warning("Index wiped. Re-upload documents.")
            except Exception as exc:
                st.error(f"Failed: {exc}")

    st.divider()
    st.markdown(
        "<div style='color:#aaa;font-size:0.75rem;'>"
        "Private Knowledge AI v1.0<br>"
        "Powered by LangChain + FAISS"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Main content ─────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🧠 Company Knowledge AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">'
    "Ask anything about company policies, procedures, and guidelines."
    "</div>",
    unsafe_allow_html=True,
)

# ── Quick-start examples ──────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💡 Example questions:")
    examples = [
        "What is the company leave policy?",
        "How many vacation days are allowed per year?",
        "What is the expense reimbursement policy?",
        "How do I request a salary advance?",
        "What are the remote work guidelines?",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, key=f"ex_{example[:20]}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

st.divider()

# ── Tabs: Chat | Retrieved Chunks ────────────────────────────
tab_chat, tab_chunks = st.tabs(["💬 Chat", "📄 Retrieved Chunks"])

with tab_chat:
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show citations under assistant messages
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander(
                        f"📚 Sources ({len(msg['citations'])} cited)", expanded=False
                    ):
                        for cit in msg["citations"]:
                            st.markdown(
                                f'<div class="citation-box">'
                                f'<strong>[{cit["index"]}]</strong> '
                                f'{cit["source"]}'
                                f'{(" — page " + str(cit["page"])) if cit.get("page") else ""}'
                                f'<br><em>{cit["preview"]}</em>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                if msg["role"] == "assistant" and msg.get("latency_ms"):
                    st.markdown(
                        f'<span class="latency-text">'
                        f'⏱ {msg["latency_ms"]:.0f} ms'
                        f'{" · ⚡ cached" if msg.get("from_cache") else ""}'
                        f"</span>",
                        unsafe_allow_html=True,
                    )

    # ── Chat input ────────────────────────────────────────────
    # Support both manual input and example button clicks
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input(
        "Ask a question about your company documents…",
        key="chat_input",
    ) or pending

    if user_input:
        if not health:
            st.error("Backend is not running. Start the API server first.")
        else:
            # Display user message immediately
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Query the backend
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("⏳ Thinking…")

                try:
                    result = api_query(
                        question=user_input,
                        session_id=st.session_state.session_id,
                        top_k=top_k,
                    )

                    answer = result["answer"]
                    citations = result.get("citations", [])
                    chunks = result.get("retrieved_chunks", [])
                    latency = result.get("latency_ms", 0)
                    from_cache = result.get("from_cache", False)

                    placeholder.markdown(answer)

                    # Citations
                    if citations:
                        with st.expander(
                            f"📚 Sources ({len(citations)} cited)", expanded=True
                        ):
                            for cit in citations:
                                st.markdown(
                                    f'<div class="citation-box">'
                                    f'<strong>[{cit["index"]}]</strong> '
                                    f'{cit["source"]}'
                                    f'{(" — page " + str(cit["page"])) if cit.get("page") else ""}'
                                    f'<br><em>{cit["preview"]}</em>'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.markdown(
                        f'<span class="latency-text">'
                        f'⏱ {latency:.0f} ms'
                        f'{" · ⚡ cached" if from_cache else ""}'
                        f"</span>",
                        unsafe_allow_html=True,
                    )

                    # Save to session
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "latency_ms": latency,
                            "from_cache": from_cache,
                        }
                    )
                    st.session_state.last_retrieval = chunks
                    st.session_state.last_citations = citations

                except httpx.HTTPStatusError as exc:
                    error_detail = exc.response.json().get("detail", str(exc))
                    placeholder.error(f"❌ API Error: {error_detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {error_detail}"}
                    )
                except Exception as exc:
                    placeholder.error(f"❌ {exc}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {exc}"}
                    )

with tab_chunks:
    if st.session_state.last_retrieval:
        st.markdown(
            f"### 🔍 Retrieved Chunks ({len(st.session_state.last_retrieval)})"
        )
        for chunk in st.session_state.last_retrieval:
            with st.expander(
                f"[{chunk['rank']}] {chunk['source']} "
                f"— score: {chunk['score']:.4f}",
                expanded=chunk["rank"] == 1,
            ):
                st.markdown(
                    f'<div class="chunk-box">{chunk["text"]}</div>',
                    unsafe_allow_html=True,
                )
                meta_cols = st.columns(3)
                meta_cols[0].caption(f"📄 {chunk['file_name']}")
                if chunk.get("page_number"):
                    meta_cols[1].caption(f"📃 Page {chunk['page_number']}")
                meta_cols[2].caption(f"🏆 Rank #{chunk['rank']}")
    else:
        st.info(
            "Retrieved document chunks will appear here after you ask a question."
        )
