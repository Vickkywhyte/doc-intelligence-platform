"""Tab 2 — Eval Dashboard: compare configs from results/runs/*.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ui.charts import make_hallucination_bar, make_heatmap, make_scatter

_RESULTS_DIR = Path("results/runs")
_API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

_DOWN_METRICS = frozenset({
    "hallucination_rate", "latency_ms_p50", "cost_per_1k_q_usd",
})

_HEADLINE_METRICS = [
    "p_at_1", "p_at_3", "p_at_5",
    "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
    "answer_accuracy", "hallucination_rate", "faithfulness",
    "refusal_correctness", "latency_ms_p50", "cost_per_1k_q_usd",
]


@st.cache_data(ttl=30)
def _load_matrix_runs() -> list[dict[str, Any]]:
    """Return a list of MatrixResult dicts, newest first."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(_RESULTS_DIR.glob("*_matrix.json"), reverse=True):
        try:
            results.append(json.loads(path.read_text()))
        except Exception:
            continue
    return results


def render_eval_tab() -> None:
    """Render Tab 2: Eval Dashboard."""
    runs = _load_matrix_runs()

    if not runs:
        _empty_state()
        return

    # Run selector
    run_labels = [
        f"{r['matrix_id']} · {r.get('n_configs_completed', '?')} configs · {r.get('started_at', '')[:10]}"
        for r in runs
    ]
    idx = st.selectbox("Select run", range(len(runs)), format_func=lambda i: run_labels[i])
    selected = runs[idx]
    entries = selected.get("configs", [])

    _summary_cards(entries, selected)
    st.divider()
    _charts_section(entries)
    st.divider()
    _comparison_table(entries)


def _empty_state() -> None:
    st.info("No evaluation runs yet. Run `make eval-matrix` and refresh.")
    if st.button("Trigger eval run"):
        import httpx
        try:
            with httpx.Client(base_url=_API_BASE, timeout=10.0) as client:
                r = client.post("/eval/run", json={})
            if r.status_code == 202:
                st.success(f"Queued run: {r.json().get('run_id')}")
            elif r.status_code == 409:
                st.warning("A run is already in progress.")
            else:
                st.error(f"Error: {r.json().get('error', {}).get('message', r.text)}")
        except Exception as exc:
            st.error(f"Could not reach API: {exc}")


def _summary_cards(entries: list[dict], matrix: dict) -> None:
    """6-card summary row."""
    def best(metric: str, lower_better: bool = False) -> tuple[float, str]:
        best_val = float("inf") if lower_better else float("-inf")
        best_code = ""
        for e in entries:
            v = e.get("aggregate_metrics", {}).get(metric)
            if v is None:
                continue
            if (lower_better and v < best_val) or (not lower_better and v > best_val):
                best_val = v
                best_code = e.get("config_code", "")
        return best_val, best_code

    acc, acc_code = best("answer_accuracy")
    p5, p5_code = best("p_at_5")
    halluc, halluc_code = best("hallucination_rate", lower_better=True)
    refusal, refusal_code = best("refusal_correctness")
    cost, cost_code = best("cost_per_1k_q_usd", lower_better=True)
    total_cost = matrix.get("total_cost_usd", 0.0)

    cols = st.columns(6)
    with cols[0]:
        st.metric("Best accuracy", f"{acc:.3f}", delta=f"{acc_code}")
    with cols[1]:
        st.metric("Best P@5", f"{p5:.3f}", delta=f"{p5_code}")
    with cols[2]:
        st.metric("Lowest halluc.", f"{halluc:.3f}", delta=f"{halluc_code}")
    with cols[3]:
        st.metric("Refusal correct.", f"{refusal:.3f}", delta=f"{refusal_code}")
    with cols[4]:
        cost_str = f"${cost:.5f}" if cost != float("inf") else "N/A"
        st.metric("Best cost/1k", cost_str, delta=f"{cost_code}")
    with cols[5]:
        st.metric("Total run cost", f"${total_cost:.4f}")


def _charts_section(entries: list[dict]) -> None:
    metric_options = [m for m in _HEADLINE_METRICS]
    metric_labels = {
        m: f"{m} {'↓' if m in _DOWN_METRICS else '↑'}" for m in metric_options
    }
    selected_metric = st.selectbox(
        "Heatmap metric",
        options=metric_options,
        format_func=lambda m: metric_labels[m],
        key="heatmap_metric",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = make_heatmap(entries, metric=selected_metric)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Heatmap data table"):
            _heatmap_table(entries, selected_metric)

    with col2:
        fig2 = make_hallucination_bar(entries)
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("Hallucination data table"):
            _halluc_table(entries)

    fig3 = make_scatter(entries)
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("Cost vs quality data table"):
        _scatter_table(entries)


def _comparison_table(entries: list[dict]) -> None:
    st.subheader("Full comparison table")

    rows = []
    for e in entries:
        code = e.get("config_code", "")
        agg = e.get("aggregate_metrics", {})
        row = {"config": code}
        for m in _HEADLINE_METRICS:
            row[m] = agg.get(m, float("nan"))
        rows.append(row)

    if not rows:
        st.caption("No data.")
        return

    df = pd.DataFrame(rows)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_bytes, "comparison.csv", "text/csv")

    st.dataframe(df, use_container_width=True)

    st.subheader("Drill-down: per-question results")
    selected_config = st.selectbox(
        "Select config to drill down",
        options=[e.get("config_code", "") for e in entries],
        key="drill_config",
    )
    selected_entry = next(
        (e for e in entries if e.get("config_code") == selected_config), None
    )
    if selected_entry:
        _drill_down(selected_entry)


def _drill_down(entry: dict) -> None:
    """Show per-question results for the selected config."""
    run_path = _RESULTS_DIR / entry.get("path", "")
    if not run_path.exists():
        st.caption(f"Run file not found: {run_path}")
        return

    try:
        run_data = json.loads(run_path.read_text())
    except Exception as exc:
        st.caption(f"Could not load run: {exc}")
        return

    per_question = run_data.get("per_question", [])
    if not per_question:
        st.caption("No per-question data.")
        return

    rows = []
    for q in per_question:
        rows.append({
            "ID": q.get("item_id", ""),
            "Question": q.get("question", "")[:80],
            "Refused": q.get("refused", False),
            "Acc": q.get("metrics", {}).get("answer_accuracy", float("nan")),
            "Halluc": q.get("metrics", {}).get("hallucination_rate", float("nan")),
            "Faith": q.get("metrics", {}).get("faithfulness", float("nan")),
            "P@5": q.get("metrics", {}).get("p_at_5", float("nan")),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    selected_q_id = st.selectbox(
        "Select question for detail",
        options=[r["ID"] for r in rows],
        key="drill_question",
    )
    q_data = next(
        (q for q in per_question if q.get("item_id") == selected_q_id), None
    )
    if q_data:
        col1, col2 = st.columns(2)
        with col1:
            hallucinated = q_data.get("hallucinated_claims", [])
            if hallucinated:
                st.markdown("**Hallucinated claims** 🔴")
                for c in hallucinated:
                    st.markdown(f"- {c}")
            else:
                st.success("No hallucinated claims detected.")
        with col2:
            st.markdown("**Judge verdict**")
            st.json(q_data.get("judge_verdict", {}))
        st.caption(f"Cost: ${q_data.get('cost_usd', 0):.6f}  Latency: {q_data.get('latency_ms', 0)} ms")


# ── Data table helpers ─────────────────────────────────────────────────────────


def _heatmap_table(entries: list[dict], metric: str) -> None:
    rows = [{"config": e.get("config_code"), metric: e.get("aggregate_metrics", {}).get(metric)} for e in entries]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _halluc_table(entries: list[dict]) -> None:
    rows = [{"config": e.get("config_code"), "hallucination_rate": e.get("aggregate_metrics", {}).get("hallucination_rate")} for e in sorted(entries, key=lambda e: e.get("aggregate_metrics", {}).get("hallucination_rate", 1.0))]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _scatter_table(entries: list[dict]) -> None:
    rows = [{"config": e.get("config_code"), "cost_per_1k_q_usd": e.get("aggregate_metrics", {}).get("cost_per_1k_q_usd"), "answer_accuracy": e.get("aggregate_metrics", {}).get("answer_accuracy")} for e in entries]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
