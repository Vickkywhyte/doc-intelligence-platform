"""Tab 1 — Chat: ask questions, see answers with citations."""

from __future__ import annotations

import os

import httpx
import streamlit as st

from ui.state import (
    CHAT_HISTORY,
    LAST_RETRIEVED,
    ChatMessage,
    add_cost,
    add_message,
    clear_chat,
)

_API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


def render_chat_tab() -> None:
    """Render Tab 1: Chat."""
    col_chat, col_sources = st.columns([2, 1])

    with col_chat:
        _header_row()
        _chat_history()
        _input_row()

    with col_sources:
        _sources_panel()


def _header_row() -> None:
    h_col, btn_col = st.columns([4, 1])
    with h_col:
        st.subheader("Ask the corpus")
    with btn_col:
        if st.button("🗑 Clear", key="clear_chat_btn"):
            clear_chat(st)
            st.rerun()


def _chat_history() -> None:
    history: list[ChatMessage] = st.session_state.get(CHAT_HISTORY, [])
    if not history:
        st.info("Ask a question about the documents in your corpus.")
        _example_chips()
        return

    for msg in history:
        if msg.role == "user":
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                if msg.refused:
                    st.markdown(
                        f"<span style='color:gray'>🚫 **I don't know** — {msg.content}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(msg.content)
                # Citation chips
                if msg.citations:
                    chips = " ".join(
                        f"`{c['chunk_id']}`" for c in msg.citations
                    )
                    st.caption(f"Sources: {chips}")
                st.caption(
                    f"{msg.model} · {msg.latency_ms} ms · ${msg.cost_usd:.5f}"
                )


def _example_chips() -> None:
    """Show example questions from golden.jsonl (factual type)."""
    examples = _load_example_questions()
    if not examples:
        return
    st.markdown("**Try a question:**")
    cols = st.columns(len(examples))
    for i, (col, q) in enumerate(zip(cols, examples)):
        with col:
            if st.button(q[:60], key=f"example_{i}"):
                st.session_state["_pending_question"] = q
                st.rerun()


def _input_row() -> None:
    pending = st.session_state.pop("_pending_question", None)
    question = st.chat_input("Ask a question…", key="chat_input")
    if pending:
        question = pending

    if question:
        add_message(st, ChatMessage(role="user", content=question))
        with st.spinner("Thinking…"):
            _call_query(question)
        st.rerun()


def _call_query(question: str) -> None:
    try:
        with httpx.Client(base_url=_API_BASE, timeout=30.0) as client:
            r = client.post("/query", json={"question": question, "return_chunks": True})
        if r.status_code != 200:
            error = r.json().get("error", {})
            st.error(f"Error {r.status_code}: {error.get('message', r.text)}")
            return
        data = r.json()
    except httpx.RequestError as exc:
        st.error(f"Could not reach API: {exc}")
        return

    msg = ChatMessage(
        role="assistant",
        content=data["answer"],
        refused=data.get("refused", False),
        model=data.get("model", ""),
        latency_ms=data.get("latency_ms", 0),
        cost_usd=data.get("cost_usd", 0.0),
        warnings=data.get("warnings", []),
        citations=data.get("citations", []),
    )
    add_message(st, msg)
    st.session_state[LAST_RETRIEVED] = data.get("retrieved", [])
    add_cost(st, data.get("cost_usd", 0.0))


def _sources_panel() -> None:
    st.subheader("Sources for last answer")
    retrieved: list[dict] = st.session_state.get(LAST_RETRIEVED, [])

    history: list[ChatMessage] = st.session_state.get(CHAT_HISTORY, [])
    last_assistant = next(
        (m for m in reversed(history) if m.role == "assistant"), None
    )

    if last_assistant and last_assistant.refused:
        st.caption("No relevant sources found")
        if retrieved:
            _render_chunk(retrieved[0], muted=True)
        return

    if not retrieved:
        st.caption("Submit a question to see source chunks.")
        return

    for chunk in retrieved:
        _render_chunk(chunk)


def _render_chunk(chunk: dict, muted: bool = False) -> None:
    score = chunk.get("score", 0.0)
    rank = chunk.get("rank", "?")
    text = chunk.get("text", "")
    page = chunk.get("page", "?")
    chunk_id = chunk.get("chunk_id", "")

    style = "color:gray;" if muted else ""
    with st.expander(f"#{rank} · score={score:.3f} · page {page}", expanded=False):
        st.markdown(f"<span style='{style}'>{text[:500]}</span>", unsafe_allow_html=True)
        st.code(chunk_id, language=None)


def _load_example_questions(max_n: int = 3) -> list[str]:
    from pathlib import Path
    import json

    golden = Path("data/eval/golden.jsonl")
    if not golden.exists():
        return []
    questions: list[str] = []
    try:
        for line in golden.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "factual":
                questions.append(item["question"])
            if len(questions) >= max_n:
                break
    except Exception:
        pass
    return questions
