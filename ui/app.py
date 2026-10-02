"""Streamlit entry point — Document Intelligence Platform UI.

Run with:
    streamlit run ui/app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="Document Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.components.chat_tab import render_chat_tab
from ui.components.eval_tab import render_eval_tab
from ui.components.runs_tab import render_runs_tab
from ui.state import (
    HOSTED_CONSENT,
    SESSION_COST,
    init_state,
)

init_state(st)


# ── Helpers (must be defined before sidebar renders) ──────────────────────────


def _read_config_summary() -> str:
    """Read configs/experiment_matrix.yaml and format a short summary."""
    try:
        import yaml
        cfg_path = Path("configs/experiment_matrix.yaml")
        if not cfg_path.exists():
            return "Config not found."
        raw = yaml.safe_load(cfg_path.read_text())
        chunking = raw.get("chunking", [{}])[0]
        embeddings = raw.get("embeddings", [{}])[0]
        retrieval = raw.get("retrieval", {})
        generation = raw.get("generation", {})
        evaluation = raw.get("evaluation", {})

        chunk_name = chunking.get("name", "?")
        chunk_size = chunking.get("size", "?")
        chunk_overlap = chunking.get("overlap", "?")
        emb_name = embeddings.get("name", "?")
        k = retrieval.get("k", "?")
        mmr = retrieval.get("mmr", False)
        model = generation.get("model", "?")
        temp = generation.get("temperature", "?")
        judge = evaluation.get("judge_model", "?")

        return (
            f"Chunking:      {chunk_name} ({chunk_size}, {chunk_overlap})\n"
            f"Embeddings:    {emb_name}\n"
            f"Retrieval:     k={k}, mmr={mmr}\n"
            f"Generator:     {model} (temp={temp})\n"
            f"Judge:         {judge} (v1)"
        )
    except Exception:
        return "Could not load config."


def _provider_badge() -> None:
    hosted = os.environ.get("USE_HOSTED", "false").lower() in ("1", "true", "yes")
    if not hosted:
        st.markdown(
            "<span style='background:#28a745;color:white;padding:2px 6px;border-radius:4px'>Provider: LOCAL</span>"
            " (no documents leave this machine)",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='background:#dc3545;color:white;padding:2px 6px;border-radius:4px'>Provider: HOSTED</span>"
            " via OpenRouter (documents sent to third parties)",
            unsafe_allow_html=True,
        )
        consent = st.checkbox(
            "I understand and consent",
            value=st.session_state.get(HOSTED_CONSENT, False),
            key="hosted_consent_checkbox",
        )
        st.session_state[HOSTED_CONSENT] = consent


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Document Intelligence Platform")
    st.divider()

    # Current config summary
    st.markdown("**Current config**")
    cfg_display = _read_config_summary()
    st.code(cfg_display, language=None)
    st.divider()

    # Session cost
    session_cost = st.session_state.get(SESSION_COST, 0.0)
    st.metric("Session cost", f"${session_cost:.5f}")
    st.divider()

    # Provider badge
    _provider_badge()
    st.divider()

    # Links
    api_url = os.environ.get("API_BASE", "http://localhost:8000")
    st.markdown(f"[FastAPI docs ↗]({api_url}/docs)")
    repo_url = os.environ.get("REPO_URL", "")
    if repo_url:
        st.markdown(f"[GitHub ↗]({repo_url})")
    with st.expander("About this project"):
        st.markdown(
            "A RAG system over company documents with a first-class evaluation pipeline. "
            "Ingests PDFs, chunks 3 ways, embeds 3 ways, retrieves, generates cited answers, "
            "and measures accuracy, hallucination rate, and faithfulness on a human-curated golden dataset."
        )


# ── Main tabs ──────────────────────────────────────────────────────────────────

tab_chat, tab_eval, tab_runs = st.tabs(["💬 Chat", "📊 Eval Dashboard", "🕐 Runs History"])

with tab_chat:
    render_chat_tab()

with tab_eval:
    render_eval_tab()

with tab_runs:
    render_runs_tab()
