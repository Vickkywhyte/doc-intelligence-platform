"""Plotly chart helpers for the Streamlit eval dashboard.

All functions are pure: (data) -> go.Figure.
Reuses figure-building logic from docintel.evaluation.report where possible.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import plotly.graph_objects as go

_DOWN_METRICS = frozenset({
    "hallucination_rate", "latency_ms_p50", "cost_per_1k_q_usd",
})


def make_heatmap(
    entries: list[dict[str, Any]],
    metric: str = "answer_accuracy",
    title: str = "Config Comparison",
) -> go.Figure:
    """Build a chunking × embedding heatmap for the given metric.

    Args:
        entries: list of MatrixConfigEntry-like dicts with
                 keys: config_code, aggregate_metrics
        metric: which metric to plot
        title: chart title
    """
    # Parse config codes into (strategy, embedder) pairs
    strategies: list[str] = []
    embedders: list[str] = []
    for e in entries:
        strat, emb = _split_code(e.get("config_code", ""))
        if strat not in strategies:
            strategies.append(strat)
        if emb not in embedders:
            embedders.append(emb)

    # Build z matrix (strategies × embedders)
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for strat in strategies:
        row_z: list[float | None] = []
        row_t: list[str] = []
        for emb in embedders:
            val = None
            for e in entries:
                s, em = _split_code(e.get("config_code", ""))
                if s == strat and em == emb:
                    val = e.get("aggregate_metrics", {}).get(metric)
                    break
            row_z.append(val if val is not None else float("nan"))
            row_t.append(f"{val:.4f}" if val is not None else "N/A")
        z.append(row_z)
        text.append(row_t)

    reverse = metric in _DOWN_METRICS
    colorscale = "RdYlGn_r" if reverse else "RdYlGn"
    direction = "↓ better" if reverse else "↑ better"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=embedders,
            y=strategies,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            reversescale=False,
            hovertemplate="<b>%{y} × %{x}</b><br>" + metric + ": %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title}<br><sup>{metric} ({direction})</sup>",
        xaxis_title="Embedding model",
        yaxis_title="Chunking strategy",
        margin={"t": 80, "b": 40, "l": 80, "r": 20},
    )
    return fig


def make_hallucination_bar(
    entries: list[dict[str, Any]],
    title: str = "Hallucination Rate by Config",
) -> go.Figure:
    """Bar chart of hallucination_rate per config, sorted ascending."""
    sorted_entries = sorted(
        entries,
        key=lambda e: e.get("aggregate_metrics", {}).get("hallucination_rate", 1.0),
    )

    codes = [e.get("config_code", "") for e in sorted_entries]
    rates = [e.get("aggregate_metrics", {}).get("hallucination_rate", 0.0) for e in sorted_entries]

    fig = go.Figure(
        data=go.Bar(
            x=codes,
            y=rates,
            marker_color="steelblue",
            hovertemplate="<b>%{x}</b><br>Hallucination rate: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Config",
        yaxis_title="Hallucination rate ↓",
        margin={"t": 60, "b": 60, "l": 60, "r": 20},
    )
    return fig


def make_scatter(
    entries: list[dict[str, Any]],
    title: str = "Cost vs Quality",
) -> go.Figure:
    """Scatter: x=cost_per_1k_q_usd, y=answer_accuracy, color by chunking."""
    strategy_colors = {"fix": "blue", "rec": "green", "sem": "orange"}

    traces: dict[str, dict[str, list]] = {}

    for e in entries:
        code = e.get("config_code", "")
        strat, emb = _split_code(code)
        agg = e.get("aggregate_metrics", {})
        cost = agg.get("cost_per_1k_q_usd", 0.0)
        acc = agg.get("answer_accuracy", 0.0)

        key = strat
        if key not in traces:
            traces[key] = {"x": [], "y": [], "text": [], "color": strategy_colors.get(strat, "gray")}
        traces[key]["x"].append(cost)
        traces[key]["y"].append(acc)
        traces[key]["text"].append(code)

    fig = go.Figure()
    for strat, td in traces.items():
        fig.add_trace(
            go.Scatter(
                x=td["x"],
                y=td["y"],
                mode="markers+text",
                text=td["text"],
                textposition="top center",
                marker={"color": td["color"], "size": 12},
                name=strat,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cost/1k: $%{x:.4f}<br>"
                    "Accuracy: %{y:.4f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Cost per 1k queries (USD) ↓",
        yaxis_title="Answer accuracy ↑",
        legend_title="Chunking",
        margin={"t": 60, "b": 60, "l": 60, "r": 20},
    )
    return fig


def _split_code(code: str) -> tuple[str, str]:
    """'fix·bge' → ('fix', 'bge')."""
    parts = code.split("·")
    if len(parts) == 2:
        return parts[0], parts[1]
    return code, ""
