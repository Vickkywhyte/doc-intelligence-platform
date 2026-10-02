"""Phase 9 — Tests for evaluation/report.py.

All tests run against the pre-built synthetic fixture at:
    tests/fixtures/synthetic_matrix/matrix.json  (6 EvalResult JSONs + 1 matrix JSON)

No network calls. No LLM calls. Plotly + kaleido are used to render PNGs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.evaluation.report import (
    ReportPaths,
    _make_heatmap,
    _make_hallucination_bar,
    _make_scatter,
    _load_entries,
    generate_report,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_matrix"
_MATRIX_PATH = _FIXTURE_DIR / "matrix.json"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_fixture_entries() -> list[dict]:
    import json
    matrix = json.loads(_MATRIX_PATH.read_text())
    return _load_entries(matrix["configs"], _FIXTURE_DIR)


# ── Fixture existence guard ────────────────────────────────────────────────────


def test_fixture_files_present() -> None:
    """All synthetic fixture files must exist before any report tests run."""
    assert _MATRIX_PATH.exists(), f"Missing fixture: {_MATRIX_PATH}"
    import json
    matrix = json.loads(_MATRIX_PATH.read_text())
    assert len(matrix["configs"]) == 6
    for cfg in matrix["configs"]:
        er_path = _FIXTURE_DIR / cfg["path"]
        assert er_path.exists(), f"Missing EvalResult fixture: {er_path}"


# ── generate_report: full pipeline ────────────────────────────────────────────


class TestGenerateReport:
    def test_returns_report_paths(self, tmp_path: Path) -> None:
        result = generate_report(str(_MATRIX_PATH), str(tmp_path))
        assert isinstance(result, ReportPaths)

    def test_all_five_files_created(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        assert Path(rp.md).exists()
        assert Path(rp.csv).exists()
        assert Path(rp.heatmap).exists()
        assert Path(rp.bar).exists()
        assert Path(rp.scatter).exists()

    def test_pngs_larger_than_1kb(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        assert Path(rp.heatmap).stat().st_size > 1024, "heatmap PNG < 1 KB"
        assert Path(rp.bar).stat().st_size > 1024, "bar PNG < 1 KB"
        assert Path(rp.scatter).stat().st_size > 1024, "scatter PNG < 1 KB"

    def test_markdown_has_six_config_rows(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        md = Path(rp.md).read_text()
        # Count table rows that start with "|" and contain a config code like "fix·bge"
        table_rows = [ln for ln in md.splitlines() if "·" in ln and ln.strip().startswith("|")]
        assert len(table_rows) == 6, f"Expected 6 config rows, got {len(table_rows)}"

    def test_markdown_contains_reproducibility_header(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        md = Path(rp.md).read_text()
        assert "Reproducibility" in md

    def test_csv_has_seven_lines(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        lines = Path(rp.csv).read_text().strip().splitlines()
        assert len(lines) == 7, f"Expected 7 CSV lines (header + 6 rows), got {len(lines)}"

    def test_csv_header_has_config_code_column(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        header = Path(rp.csv).read_text().splitlines()[0]
        assert "config_code" in header

    def test_matrix_id_in_markdown(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        md = Path(rp.md).read_text()
        assert "20261002T000000_matrix" in md

    def test_output_paths_under_output_dir(self, tmp_path: Path) -> None:
        rp = generate_report(str(_MATRIX_PATH), str(tmp_path))
        for p in (rp.md, rp.csv, rp.heatmap, rp.bar, rp.scatter):
            assert str(tmp_path) in str(p), f"{p} not under {tmp_path}"


# ── Heatmap colorscale direction ───────────────────────────────────────────────


class TestHeatmapColorscale:
    def test_up_metric_not_reversed(self) -> None:
        """answer_accuracy is ↑; reversescale must be False."""
        entries = _load_fixture_entries()
        fig = _make_heatmap(entries, "answer_accuracy", "test_matrix")
        assert fig.data[0].reversescale is False or fig.data[0].reversescale is None

    def test_down_metric_reversed(self) -> None:
        """hallucination_rate is ↓; reversescale must be True."""
        entries = _load_fixture_entries()
        fig = _make_heatmap(entries, "hallucination_rate", "test_matrix")
        assert fig.data[0].reversescale is True

    def test_latency_reversed(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_heatmap(entries, "latency_ms_p50", "test_matrix")
        assert fig.data[0].reversescale is True

    def test_cost_reversed(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_heatmap(entries, "cost_per_1k_q_usd", "test_matrix")
        assert fig.data[0].reversescale is True

    def test_heatmap_has_6_cells(self) -> None:
        """3 strategies × 2 embedders = 6 cells in the z matrix."""
        entries = _load_fixture_entries()
        fig = _make_heatmap(entries, "answer_accuracy", "test_matrix")
        z = fig.data[0].z
        total_cells = sum(len(row) for row in z)
        assert total_cells == 6

    def test_heatmap_title_contains_matrix_id(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_heatmap(entries, "mrr", "my_matrix_id")
        assert "my_matrix_id" in fig.layout.title.text


# ── Hallucination bar ─────────────────────────────────────────────────────────


class TestHallucinationBar:
    def test_bar_has_6_entries(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_hallucination_bar(entries, "test_matrix")
        assert len(fig.data[0].x) == 6

    def test_bar_values_non_negative(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_hallucination_bar(entries, "test_matrix")
        for v in fig.data[0].y:
            assert v >= 0.0

    def test_bar_sorted_ascending(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_hallucination_bar(entries, "test_matrix")
        vals = list(fig.data[0].y)
        assert vals == sorted(vals), "Bar chart should be sorted ascending"


# ── Scatter ────────────────────────────────────────────────────────────────────


class TestScatter:
    def test_scatter_traces_cover_all_configs(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_scatter(entries, "test_matrix")
        total_points = sum(len(t.x) for t in fig.data)
        assert total_points == 6

    def test_scatter_title_contains_matrix_id(self) -> None:
        entries = _load_fixture_entries()
        fig = _make_scatter(entries, "my_id")
        assert "my_id" in fig.layout.title.text


# ── EvalResult fixtures: schema completeness ───────────────────────────────────


class TestFixtureCompleteness:
    def test_sixteen_aggregate_metrics_present(self) -> None:
        """Every fixture EvalResult must have all 16 expected aggregate metrics."""
        expected = {
            "p_at_1", "p_at_3", "p_at_5",
            "recall_at_1", "recall_at_3", "recall_at_5",
            "mrr",
            "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
            "answer_accuracy", "hallucination_rate",
            "faithfulness", "refusal_correctness",
            "latency_ms_p50", "cost_per_1k_q_usd",
        }
        entries = _load_fixture_entries()
        for entry in entries:
            missing = expected - set(entry["aggregate_metrics"].keys())
            assert not missing, (
                f"Config {entry['config_code']} missing metrics: {missing}"
            )

    def test_three_per_question_entries(self) -> None:
        import json
        for cfg_file in _FIXTURE_DIR.glob("run_*.json"):
            er = json.loads(cfg_file.read_text())
            assert len(er["per_question"]) == 3, (
                f"{cfg_file.name}: expected 3 per_question entries"
            )

    def test_matrix_references_six_configs(self) -> None:
        import json
        matrix = json.loads(_MATRIX_PATH.read_text())
        assert matrix["n_configs_total"] == 6
        assert matrix["n_configs_completed"] == 6
        assert len(matrix["configs"]) == 6
