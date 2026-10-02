"""Tab 3 — Runs History: table of past experiment runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

_RESULTS_DIR = Path("results/runs")
_REPORTS_DIR = Path("results/reports")
_API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


@st.cache_data(ttl=30)
def _load_runs() -> list[dict[str, Any]]:
    """Scan results/runs/*_matrix.json, newest first."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(_RESULTS_DIR.glob("*_matrix.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            data["_matrix_path"] = str(path)
            runs.append(data)
        except Exception:
            continue
    return runs


def render_runs_tab() -> None:
    """Render Tab 3: Runs History."""
    runs = _load_runs()

    if not runs:
        st.info("No runs yet. The first matrix run will appear here.")
        return

    rows = []
    for r in runs:
        configs = r.get("configs", [])
        first = configs[0] if configs else {}
        code = first.get("config_code", "")
        rows.append({
            "Run ID": r.get("matrix_id", ""),
            "Configs": r.get("n_configs_completed", 0),
            "Config code (first)": code,
            "Started": r.get("started_at", "")[:19].replace("T", " "),
            "Cost ($)": r.get("total_cost_usd", 0.0),
            "Status": r.get("status", ""),
        })

    df = pd.DataFrame(rows)
    st.subheader("All runs")

    # Status filter
    status_filter = st.selectbox(
        "Filter by status",
        options=["all", "complete", "aborted"],
        key="runs_status_filter",
    )
    if status_filter != "all":
        df = df[df["Status"] == status_filter]

    st.dataframe(df, use_container_width=True)

    # Detail panel for selected run
    st.subheader("Run actions")
    run_ids = [r.get("matrix_id", "") for r in runs]
    selected_run_id = st.selectbox("Select run", options=run_ids, key="selected_run_detail")

    selected_run = next((r for r in runs if r.get("matrix_id") == selected_run_id), None)
    if not selected_run:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Open in Dashboard", key="open_dashboard_btn"):
            st.session_state["active_tab"] = 1
            st.session_state["selected_run"] = selected_run_id
            st.rerun()

    with col2:
        _download_report_btn(selected_run_id)

    with col3:
        _download_json_btn(selected_run)

    with col4:
        if st.button("Re-run", key="rerun_btn"):
            _trigger_rerun(selected_run)

    # Show summary for selected run
    with st.expander("Run details", expanded=False):
        st.json({
            "matrix_id": selected_run.get("matrix_id"),
            "status": selected_run.get("status"),
            "n_configs_completed": selected_run.get("n_configs_completed"),
            "total_cost_usd": selected_run.get("total_cost_usd"),
            "started_at": selected_run.get("started_at"),
            "ended_at": selected_run.get("ended_at"),
        })

        configs_df = pd.DataFrame([
            {
                "config_code": c.get("config_code"),
                "run_id": c.get("run_id"),
                **{k: round(v, 4) for k, v in c.get("aggregate_metrics", {}).items()},
            }
            for c in selected_run.get("configs", [])
        ])
        if not configs_df.empty:
            st.dataframe(configs_df, use_container_width=True)


def _download_report_btn(run_id: str) -> None:
    md_path = _REPORTS_DIR / f"{run_id}.md"
    if md_path.exists():
        st.download_button(
            "Download report",
            data=md_path.read_text(encoding="utf-8"),
            file_name=f"{run_id}_report.md",
            mime="text/markdown",
            key=f"dl_report_{run_id}",
        )
    else:
        if st.button("Download report", key=f"dl_report_generate_{run_id}"):
            try:
                with httpx.Client(base_url=_API_BASE, timeout=60.0) as client:
                    r = client.get(f"/eval/runs/{run_id}/report")
                if r.status_code == 200:
                    st.download_button(
                        "Save report",
                        data=r.text,
                        file_name=f"{run_id}_report.md",
                        mime="text/markdown",
                        key=f"dl_report_save_{run_id}",
                    )
                else:
                    st.error(r.json().get("error", {}).get("message", "Error generating report"))
            except Exception as exc:
                st.error(f"Could not fetch report: {exc}")


def _download_json_btn(run: dict) -> None:
    run_id = run.get("matrix_id", "")
    run_copy = {k: v for k, v in run.items() if k != "_matrix_path"}
    st.download_button(
        "Download JSON",
        data=json.dumps(run_copy, indent=2),
        file_name=f"{run_id}_matrix.json",
        mime="application/json",
        key=f"dl_json_{run_id}",
    )


def _trigger_rerun(run: dict) -> None:
    """Write a replay config and POST /eval/run."""
    run_id = run.get("matrix_id", "")
    replay_path = Path(f"configs/replay_{run_id}.yaml")
    try:
        import yaml
        src = Path("configs/experiment_matrix.yaml")
        if src.exists():
            replay_path.write_text(src.read_text())
        else:
            st.error("Cannot find source config to replay.")
            return

        with httpx.Client(base_url=_API_BASE, timeout=10.0) as client:
            r = client.post("/eval/run", json={"config_path": str(replay_path), "notes": f"replay of {run_id}"})
        if r.status_code == 202:
            st.success(f"Re-run queued: {r.json().get('run_id')}")
        elif r.status_code == 409:
            st.warning("A run is already in progress.")
        else:
            st.error(r.json().get("error", {}).get("message", "Error starting re-run"))
    except Exception as exc:
        st.error(f"Re-run failed: {exc}")
