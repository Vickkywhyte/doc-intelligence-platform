"""Session state keys and helpers for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Key constants ──────────────────────────────────────────────────────────────

CHAT_HISTORY = "chat_history"
SESSION_COST = "session_cost"
LAST_RETRIEVED = "last_retrieved"
ACTIVE_TAB = "active_tab"
SELECTED_RUN = "selected_run"
HOSTED_CONSENT = "hosted_consent"


# ── Message models (not Pydantic — stays in Streamlit layer) ──────────────────


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    refused: bool = False
    model: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────


def init_state(st) -> None:
    """Initialise all session-state keys with defaults on first run."""
    defaults: dict[str, Any] = {
        CHAT_HISTORY: [],
        SESSION_COST: 0.0,
        LAST_RETRIEVED: [],
        ACTIVE_TAB: 0,
        SELECTED_RUN: None,
        HOSTED_CONSENT: False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def add_message(st, msg: ChatMessage) -> None:
    st.session_state[CHAT_HISTORY].append(msg)


def clear_chat(st) -> None:
    st.session_state[CHAT_HISTORY] = []
    st.session_state[LAST_RETRIEVED] = []
    st.session_state[SESSION_COST] = 0.0


def add_cost(st, cost_usd: float) -> None:
    st.session_state[SESSION_COST] = round(
        st.session_state.get(SESSION_COST, 0.0) + cost_usd, 6
    )
