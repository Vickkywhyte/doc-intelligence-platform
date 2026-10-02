"""Report generator for experiment matrix results — ROADMAP Phase 9.

Public API:
    generate_report(matrix_path, output_dir, *, metric) -> ReportPaths

Reads a MatrixResult JSON, loads each referenced EvalResult, and writes:
    results/reports/{matrix_id}.md
    results/reports/{matrix_id}.csv
    results/figures/{matrix_id}_heatmap.png
    results/figures/{matrix_id}_hallucination_bar.png
    results/figures/{matrix_id}_cost_quality_scatter.png

See 06_EVAL_SPEC §6.1, 10_OBSERVABILITY §9, ROADMAP Phase 9.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import structlog
from pydantic import BaseModel

log = structlog.get_logger()

# Metrics where lower is better (reversed colorscale for heatmap)
_DOWN_METRICS: frozenset[str] = frozenset(
    {"hallucination_rate", "latency_ms_p50", "cost_per_1k_q_usd"}
)

# Ordered list of aggregate metrics for the headline table (§6.1)
_HEADLINE_METRICS: list[str] = [
    "p_at_1", "p_at_3", "p_at_5",
    "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr",
    "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
    "answer_accuracy",
    "hallucination_rate",
    "faithfulness",
    "refusal_correctness",
    "latency_ms_p50",
    "cost_per_1k_q_usd",
]


# ── Pydantic model ─────────────────────────────────────────────────────────────


class ReportPaths(BaseModel):
    """Paths to all generated report artifacts."""

    model_config = {"extra": "ignore"}

    md: Path
    csv: Path
    heatmap: Path
    bar: Path
    scatter: Path


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_report(
    matrix_path: str,
    output_dir: str,
    *,
    metric: str = "answer_accuracy",
) -> ReportPaths:
    """Generate Markdown + CSV + 3 PNG figures for a matrix result.

    Args:
        matrix_path: Path to the MatrixResult JSON.
        output_dir:  Base output directory (e.g. "results/").
        metric:      Metric to use for the heatmap (default "answer_accuracy").

    Returns:
        ReportPaths with absolute paths to all 5 output files.
    """
    matrix = _load_json(matrix_path)
    matrix_id = matrix["matrix_id"]
    matrix_dir = Path(matrix_path).parent

    # Load all EvalResult JSONs
    entries = _load_entries(matrix["configs"], matrix_dir)

    # Create output directories
    report_dir = Path(output_dir) / "reports"
    figure_dir = Path(output_dir) / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    heatmap_path = figure_dir / f"{matrix_id}_heatmap.png"
    bar_path = figure_dir / f"{matrix_id}_hallucination_bar.png"
    scatter_path = figure_dir / f"{matrix_id}_cost_quality_scatter.png"

    heatmap_fig = _make_heatmap(entries, metric, matrix_id)
    bar_fig = _make_hallucination_bar(entries, matrix_id)
    scatter_fig = _make_scatter(entries, matrix_id)

    heatmap_fig.write_image(str(heatmap_path))
    bar_fig.write_image(str(bar_path))
    scatter_fig.write_image(str(scatter_path))

    log.info("report_figures_written", heatmap=str(heatmap_path), bar=str(bar_path), scatter=str(scatter_path))

    # Generate Markdown
    md_path = report_dir / f"{matrix_id}.md"
    md_content = _build_markdown(matrix, entries, matrix_id, heatmap_path, bar_path, scatter_path)
    md_path.write_text(md_content, encoding="utf-8")

    # Generate CSV
    csv_path = report_dir / f"{matrix_id}.csv"
    _write_csv(entries, csv_path)

    log.info("report_written", md=str(md_path), csv=str(csv_path), matrix_id=matrix_id)

    return ReportPaths(
        md=md_path,
        csv=csv_path,
        heatmap=heatmap_path,
        bar=bar_path,
        scatter=scatter_path,
    )


# ── Figure builders ────────────────────────────────────────────────────────────


def _make_heatmap(
    entries: list[dict[str, Any]],
    metric: str,
    matrix_id: str,
) -> go.Figure:
    """Build a heatmap: rows=chunking, cols=embedding, values=metric.

    Uses RdYlGn colorscale; reversed for ↓ metrics (hallucination, latency, cost).
    Returns a Plotly Figure (exported to PNG by the caller).
    """
    strategies: list[str] = []
    embedders: list[str] = []
    for e in entries:
        s, emb = _parse_config_code(e["config_code"])
        if s not in strategies:
            strategies.append(s)
        if emb not in embedders:
            embedders.append(emb)

    n_rows = len(strategies)
    n_cols = len(embedders)
    z: list[list[float | None]] = [[None] * n_cols for _ in range(n_rows)]
    text: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]

    for e in entries:
        s, emb = _parse_config_code(e["config_code"])
        val = e["aggregate_metrics"].get(metric)
        if val is not None and s in strategies and emb in embedders:
            ri = strategies.index(s)
            ci = embedders.index(emb)
            z[ri][ci] = float(val)
            text[ri][ci] = f"{val:.3f}"

    is_down = metric in _DOWN_METRICS

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=embedders,
            y=strategies,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            reversescale=is_down,
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"{matrix_id} — {metric}",
        xaxis_title="Embedding",
        yaxis_title="Chunking",
        height=350,
        width=600,
    )
    return fig


def _make_hallucination_bar(
    entries: list[dict[str, Any]],
    matrix_id: str,
) -> go.Figure:
    """Bar chart: hallucination_rate per config, sorted ascending, with 95% CI error bars."""
    sorted_entries = sorted(
        entries,
        key=lambda e: float(e["aggregate_metrics"].get("hallucination_rate", 0.0)),
    )

    labels = [e["config_code"] for e in sorted_entries]
    values = [float(e["aggregate_metrics"].get("hallucination_rate", 0.0)) for e in sorted_entries]

    # Error bars from aggregate_ci if available
    errors: list[float] = []
    for e in sorted_entries:
        ci = e.get("aggregate_ci", {}).get("hallucination_rate")
        if ci and len(ci) == 2:
            from docintel.evaluation.stats import ci_half_width
            errors.append(ci_half_width(float(ci[0]), float(ci[1])))
        else:
            errors.append(0.0)

    error_y: dict[str, Any] | None = None
    if any(v > 0 for v in errors):
        error_y = {"type": "data", "array": errors, "visible": True}

    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=values,
            error_y=error_y,
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title=f"{matrix_id} — hallucination_rate ↓",
        xaxis_title="Config",
        yaxis_title="Hallucination Rate",
        height=400,
        width=700,
    )
    return fig


def _make_scatter(
    entries: list[dict[str, Any]],
    matrix_id: str,
) -> go.Figure:
    """Scatter: x=cost_per_1k_q_usd, y=answer_accuracy; color=chunking, symbol=embedding."""
    # Assign colours to unique strategies and symbols to unique embedders
    strategies: list[str] = []
    embedders: list[str] = []
    for e in entries:
        s, emb = _parse_config_code(e["config_code"])
        if s not in strategies:
            strategies.append(s)
        if emb not in embedders:
            embedders.append(emb)

    _COLORS = ["royalblue", "tomato", "mediumseagreen", "orange", "mediumpurple"]
    _SYMBOLS = ["circle", "square", "diamond", "cross", "x"]

    # Collect latency values for normalised size
    latencies = [
        float(e["aggregate_metrics"].get("latency_ms_p50", 1000.0))
        for e in entries
    ]
    lat_min = min(latencies) if latencies else 1.0
    lat_max = max(latencies) if latencies else 1.0
    lat_range = lat_max - lat_min if lat_max != lat_min else 1.0

    traces: dict[str, go.Scatter] = {}

    for e in entries:
        s, emb = _parse_config_code(e["config_code"])
        x = float(e["aggregate_metrics"].get("cost_per_1k_q_usd", 0.0))
        y = float(e["aggregate_metrics"].get("answer_accuracy", 0.0))
        lat = float(e["aggregate_metrics"].get("latency_ms_p50", 1000.0))
        size = 10 + 15 * (lat - lat_min) / lat_range

        color = _COLORS[strategies.index(s) % len(_COLORS)]
        symbol = _SYMBOLS[embedders.index(emb) % len(_SYMBOLS)]
        label = e["config_code"]

        trace_key = s
        if trace_key not in traces:
            traces[trace_key] = go.Scatter(
                x=[], y=[], mode="markers+text",
                name=s,
                marker={"color": color, "symbol": symbol, "size": []},  # type: ignore[arg-type]
                text=[], textposition="top center",
            )

        t = traces[trace_key]
        t.x = list(t.x) + [x]  # type: ignore[operator]
        t.y = list(t.y) + [y]  # type: ignore[operator]
        t.text = list(t.text) + [label]  # type: ignore[operator]
        t.marker["size"] = list(t.marker["size"]) + [size]  # type: ignore[index]

    fig = go.Figure(data=list(traces.values()))
    fig.update_layout(
        title=f"{matrix_id} — cost vs quality",
        xaxis_title="Cost per 1k questions (USD)",
        yaxis_title="Answer Accuracy",
        height=500,
        width=700,
    )
    return fig


# ── Markdown builder ───────────────────────────────────────────────────────────


def _build_markdown(
    matrix: dict[str, Any],
    entries: list[dict[str, Any]],
    matrix_id: str,
    heatmap_path: Path,
    bar_path: Path,
    scatter_path: Path,
) -> str:
    lines: list[str] = []

    lines.append(f"# Experiment Matrix {matrix_id}\n")

    # Reproducibility receipt
    first_er = entries[0]["eval_result"] if entries else {}
    lines.append("## Reproducibility receipt\n")
    lines.append(_build_receipt(matrix, first_er))
    lines.append("")

    # Headline table
    lines.append("## Headline table\n")
    lines.append(_build_headline_table(entries))
    lines.append("")

    # Narrative
    lines.append("## Narrative\n")
    lines.append(_build_narrative(entries))
    lines.append("")

    # Figure references
    lines.append("## Figure references\n")
    lines.append(f"- Heatmap: `{heatmap_path}`")
    lines.append(f"- Hallucination bar: `{bar_path}`")
    lines.append(f"- Cost–quality scatter: `{scatter_path}`")
    lines.append("")

    return "\n".join(lines)


def _build_receipt(matrix: dict[str, Any], first_er: dict[str, Any]) -> str:
    config = first_er.get("config", {})
    gen_cfg = config.get("generation", {})
    receipt_lines = [
        f"- Run ID:            {matrix.get('matrix_id', 'n/a')}",
        f"- Config hash:       {first_er.get('config_hash', 'n/a')}",
        f"- Git SHA:           {first_er.get('git_sha', 'n/a')}",
        f"- Python:            {first_er.get('python_version', 'n/a')}",
        f"- Judge:             {first_er.get('judge_model', 'n/a')} "
        f"(prompt {first_er.get('judge_prompt_version', 'n/a')}, temp=0)",
        f"- Generator:         {gen_cfg.get('model', 'n/a')} "
        f"(temp={gen_cfg.get('temperature', 'n/a')})",
        f"- Seed:              {first_er.get('seed', 'n/a')}",
        f"- Nondeterminism:    {first_er.get('notes', 'none declared') or 'none declared'}",
        f"- Started:           {matrix.get('started_at', 'n/a')}",
        f"- Ended:             {matrix.get('ended_at', 'n/a')}",
        f"- Total cost:        ${matrix.get('total_cost_usd', 0.0):.4f}",
    ]
    return "\n".join(receipt_lines)


def _build_headline_table(entries: list[dict[str, Any]]) -> str:
    # Collect metric names actually present across entries
    present_metrics = [
        m for m in _HEADLINE_METRICS
        if any(m in e["aggregate_metrics"] for e in entries)
    ]

    header_cols = ["Config"] + [_metric_label(m) for m in present_metrics]
    separator = ["---"] * len(header_cols)

    rows = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for e in entries:
        agg = e["aggregate_metrics"]
        ci_map = e.get("aggregate_ci", {})
        cells = [e["config_code"]]
        for m in present_metrics:
            val = agg.get(m)
            if val is None:
                cells.append("—")
            else:
                ci = ci_map.get(m)
                if ci and len(ci) == 2:
                    from docintel.evaluation.stats import ci_half_width
                    hw = ci_half_width(float(ci[0]), float(ci[1]))
                    cells.append(f"{val:.3f} (±{hw:.3f})")
                else:
                    cells.append(f"{val:.3f}")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def _build_narrative(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No configs completed."

    def _get(e: dict, k: str, default: float = 0.0) -> float:
        return float(e["aggregate_metrics"].get(k, default))

    best = max(entries, key=lambda e: _get(e, "answer_accuracy"))
    worst = min(entries, key=lambda e: _get(e, "answer_accuracy"))
    costliest = max(entries, key=lambda e: _get(e, "cost_per_1k_q_usd"))
    cheapest = min(entries, key=lambda e: _get(e, "cost_per_1k_q_usd"))
    fastest = min(entries, key=lambda e: _get(e, "latency_ms_p50", 1e9))

    return (
        f"- Best answer accuracy: **{best['config_code']}** "
        f"({_get(best, 'answer_accuracy'):.3f}).\n"
        f"- Worst answer accuracy: **{worst['config_code']}** "
        f"({_get(worst, 'answer_accuracy'):.3f}).\n"
        f"- Most expensive: **{costliest['config_code']}** "
        f"(${_get(costliest, 'cost_per_1k_q_usd'):.4f}/1k q).\n"
        f"- Cheapest: **{cheapest['config_code']}** "
        f"(${_get(cheapest, 'cost_per_1k_q_usd'):.4f}/1k q).\n"
        f"- Fastest: **{fastest['config_code']}** "
        f"({_get(fastest, 'latency_ms_p50'):.0f} ms p50)."
    )


# ── CSV writer ─────────────────────────────────────────────────────────────────


def _write_csv(entries: list[dict[str, Any]], path: Path) -> None:
    if not entries:
        return

    present_metrics = [
        m for m in _HEADLINE_METRICS
        if any(m in e["aggregate_metrics"] for e in entries)
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config_code", "run_id"] + present_metrics)
        for e in entries:
            agg = e["aggregate_metrics"]
            row = [e["config_code"], e.get("run_id", "")]
            for m in present_metrics:
                val = agg.get(m)
                row.append(f"{val:.6f}" if val is not None else "")
            writer.writerow(row)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_path(raw: str, matrix_path: str) -> Path:
    """Resolve an EvalResult path relative to the matrix JSON's directory."""
    p = Path(raw)
    if p.is_absolute():
        return p
    # Resolve relative to matrix file's parent
    return (Path(matrix_path).parent / p).resolve()


def _load_entries(
    configs: list[dict[str, Any]],
    matrix_dir: Path,
) -> list[dict[str, Any]]:
    entries = []
    for cfg in configs:
        raw_path = cfg["path"]
        p = Path(raw_path)
        if not p.is_absolute():
            p = (matrix_dir / p).resolve()

        eval_result: dict[str, Any] = {}
        if p.exists():
            eval_result = _load_json(p)

        entries.append(
            {
                "config_code": cfg["config_code"],
                "run_id": cfg.get("run_id", ""),
                "aggregate_metrics": dict(cfg.get("aggregate_metrics", {})),
                "aggregate_ci": eval_result.get("aggregate_ci", {}),
                "eval_result": eval_result,
            }
        )
    return entries


def _parse_config_code(code: str) -> tuple[str, str]:
    """Split 'fix·bge' → ('fix', 'bge'). Falls back to (code, '') on error."""
    parts = code.split("·")
    if len(parts) == 2:
        return parts[0], parts[1]
    return code, ""


def _metric_label(metric: str) -> str:
    """Short display name for a metric key."""
    _MAP = {
        "p_at_1": "P@1", "p_at_3": "P@3", "p_at_5": "P@5",
        "recall_at_1": "R@1", "recall_at_3": "R@3", "recall_at_5": "R@5",
        "mrr": "MRR",
        "ndcg_at_1": "NDCG@1", "ndcg_at_3": "NDCG@3", "ndcg_at_5": "NDCG@5",
        "answer_accuracy": "Acc",
        "hallucination_rate": "Halluc ↓",
        "faithfulness": "Faith",
        "refusal_correctness": "Refusal",
        "latency_ms_p50": "Latency ↓",
        "cost_per_1k_q_usd": "$/1k ↓",
    }
    return _MAP.get(metric, metric)
