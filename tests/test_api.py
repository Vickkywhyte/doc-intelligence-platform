"""Tests for the FastAPI API layer.

Uses FastAPI TestClient (synchronous, no async required).
All tests run without an OpenRouter key (USE_HOSTED=false).

Route registration check: tests deliberately assert HTTP 200/202/400/404
on real endpoint paths — any missing route would return 404 and fail the
assertion, making the tests self-proving.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("USE_HOSTED", "false")

# ── Fixtures ───────────────────────────────────────────────────────────────────

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_matrix"
_MATRIX_ID = "20261002T000000_matrix"


@pytest.fixture(scope="session")
def app():
    """Single shared app instance (avoids repeated import overhead)."""
    from api.main import app as _app
    return _app


@pytest.fixture()
def client(app):
    """FastAPI TestClient — routes are live, no network calls made."""
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def runs_dir(tmp_path):
    """Temp results/runs dir pre-populated with synthetic_matrix fixtures.

    File layout:
      {tmp}/runs/{MATRIX_ID}_matrix.json   <- matrix result
      {tmp}/runs/run_fix_bge.json          <- per-config eval results
      {tmp}/runs/run_fix_min.json
      ... (all 6 configs)
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    # Copy matrix.json → {matrix_id}_matrix.json
    src_matrix = _FIXTURE_DIR / "matrix.json"
    dst_matrix = runs / f"{_MATRIX_ID}_matrix.json"
    shutil.copy(src_matrix, dst_matrix)
    # Copy all per-config run files (referenced by relative path in matrix.json)
    for src in _FIXTURE_DIR.glob("run_*.json"):
        shutil.copy(src, runs / src.name)
    return runs


@pytest.fixture()
def reports_dir(tmp_path, runs_dir):
    """Temp results/reports dir with a pre-generated markdown report."""
    reports = tmp_path / "reports"
    reports.mkdir()
    md = reports / f"{_MATRIX_ID}.md"
    md.write_text(
        f"# Report {_MATRIX_ID}\n\n## Reproducibility\n\nmatrix_id: {_MATRIX_ID}\n\n"
        "## Headline metrics\n\n| config | answer_accuracy |\n|---|---|\n| fix·bge | 0.8200 |\n"
    )
    return reports


@pytest.fixture()
def figures_dir(tmp_path):
    """Temp results/figures dir with placeholder PNG files."""
    figs = tmp_path / "figures"
    figs.mkdir()
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    for name in ("heatmap", "hallucination_bar", "cost_quality_scatter"):
        (figs / f"{_MATRIX_ID}_{name}.png").write_bytes(png_header)
    return figs


# ── Route registration smoke-test ──────────────────────────────────────────────


class TestRouteRegistration:
    """Verify all 8 API routes are actually registered.

    These tests would return 405 or 404 (not our custom codes) if a route
    were missing, causing assertion failures on status codes or error codes.
    """

    def test_all_expected_routes_in_openapi(self, app) -> None:
        """OpenAPI spec must list all 8 paths from 07_API_SPEC.md §2."""
        paths = set(app.openapi()["paths"].keys())
        expected = {
            "/health",
            "/ingest",
            "/query",
            "/eval/run",
            "/eval/runs",
            "/eval/runs/{run_id}",
            "/eval/runs/{run_id}/report",
            "/eval/runs/{run_id}/figures/{name}",
        }
        missing = expected - paths
        assert not missing, f"Missing routes from OpenAPI spec: {missing}"

    def test_health_is_get(self, app) -> None:
        methods = app.openapi()["paths"]["/health"]
        assert "get" in methods

    def test_query_is_post(self, app) -> None:
        methods = app.openapi()["paths"]["/query"]
        assert "post" in methods

    def test_ingest_is_post(self, app) -> None:
        methods = app.openapi()["paths"]["/ingest"]
        assert "post" in methods

    def test_eval_run_is_post(self, app) -> None:
        methods = app.openapi()["paths"]["/eval/run"]
        assert "post" in methods

    def test_eval_runs_is_get(self, app) -> None:
        methods = app.openapi()["paths"]["/eval/runs"]
        assert "get" in methods


# ── GET /health ────────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_200(self, client) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_has_required_fields(self, client) -> None:
        body = client.get("/health").json()
        for field in ("status", "version", "git_sha", "python_version", "uptime_s"):
            assert field in body, f"Missing field: {field}"

    def test_uptime_non_negative(self, client) -> None:
        assert client.get("/health").json()["uptime_s"] >= 0

    def test_request_id_header_set(self, client) -> None:
        r = client.get("/health")
        assert "x-request-id" in r.headers


# ── POST /ingest ───────────────────────────────────────────────────────────────


class TestIngest:
    def test_no_files_returns_400_or_422(self, client) -> None:
        r = client.post("/ingest", data={"doc_type": "contract"})
        assert r.status_code in (400, 422)

    def test_invalid_doc_type_returns_400(self, client) -> None:
        content = b"%PDF-1.4 fake"
        r = client.post(
            "/ingest",
            files=[("files", ("test.pdf", content, "application/pdf"))],
            data={"doc_type": "invalid_type"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_DOC_TYPE"

    def test_invalid_doc_type_envelope_shape(self, client) -> None:
        r = client.post(
            "/ingest",
            files=[("files", ("t.pdf", b"%PDF fake", "application/pdf"))],
            data={"doc_type": "bogus"},
        )
        body = r.json()
        assert "error" in body
        for field in ("code", "message", "detail", "request_id"):
            assert field in body["error"]

    def test_valid_pdf_attempt_returns_200_or_422(self, client) -> None:
        content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        r = client.post(
            "/ingest",
            files=[("files", ("minimal.pdf", content, "application/pdf"))],
            data={"doc_type": "other"},
        )
        assert r.status_code in (200, 422)

    def test_200_response_has_processed_failed_summary(self, client) -> None:
        content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        r = client.post(
            "/ingest",
            files=[("files", ("minimal.pdf", content, "application/pdf"))],
            data={"doc_type": "other"},
        )
        if r.status_code == 200:
            body = r.json()
            assert "processed" in body
            assert "failed" in body
            assert "summary" in body


# ── POST /query ────────────────────────────────────────────────────────────────


class TestQuery:
    def test_valid_question_returns_200(self, client) -> None:
        r = client.post("/query", json={"question": "What is in the documents?"})
        assert r.status_code == 200

    def test_response_has_required_fields(self, client) -> None:
        r = client.post("/query", json={"question": "What are the payment terms?"})
        body = r.json()
        for field in ("query", "answer", "refused", "citations", "retrieved", "model", "latency_ms", "cost_usd", "warnings"):
            assert field in body, f"Missing field: {field}"

    def test_query_echoed_in_response(self, client) -> None:
        question = "What is the notice period?"
        assert client.post("/query", json={"question": question}).json()["query"] == question

    def test_refused_is_bool(self, client) -> None:
        r = client.post("/query", json={"question": "Something unrelated?"})
        assert isinstance(r.json()["refused"], bool)

    def test_citations_is_list(self, client) -> None:
        assert isinstance(client.post("/query", json={"question": "test"}).json()["citations"], list)

    def test_retrieved_is_list(self, client) -> None:
        assert isinstance(client.post("/query", json={"question": "test"}).json()["retrieved"], list)

    def test_empty_question_returns_400(self, client) -> None:
        r = client.post("/query", json={"question": "   "})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "EMPTY_QUESTION"

    def test_empty_string_question_returns_400(self, client) -> None:
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 400

    def test_question_too_long_returns_400(self, client) -> None:
        r = client.post("/query", json={"question": "x" * 2001})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "QUESTION_TOO_LONG"

    def test_return_chunks_false_returns_empty_retrieved(self, client) -> None:
        r = client.post("/query", json={"question": "test question", "return_chunks": False})
        assert r.status_code == 200
        assert r.json()["retrieved"] == []

    def test_error_envelope_has_request_id(self, client) -> None:
        r = client.post("/query", json={"question": ""})
        assert "request_id" in r.json()["error"]


# ── POST /eval/run ─────────────────────────────────────────────────────────────


class TestEvalRun:
    def test_missing_config_returns_404(self, client) -> None:
        r = client.post("/eval/run", json={"config_path": "configs/does_not_exist.yaml"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "CONFIG_NOT_FOUND"

    def test_valid_config_returns_202_or_409(self, client) -> None:
        r = client.post("/eval/run", json={"config_path": "configs/experiment_matrix.yaml"})
        assert r.status_code in (202, 409)

    def test_202_response_has_required_fields(self, client) -> None:
        r = client.post("/eval/run", json={"config_path": "configs/experiment_matrix.yaml"})
        if r.status_code == 202:
            body = r.json()
            for field in ("run_id", "status", "n_configs", "n_questions", "estimated_cost_usd", "estimated_duration_s"):
                assert field in body, f"Missing field: {field}"

    def test_202_status_is_queued(self, client) -> None:
        r = client.post("/eval/run", json={"config_path": "configs/experiment_matrix.yaml"})
        if r.status_code == 202:
            assert r.json()["status"] == "queued"

    def test_409_error_code_run_in_progress(self, client) -> None:
        r = client.post("/eval/run", json={"config_path": "configs/experiment_matrix.yaml"})
        if r.status_code == 409:
            assert r.json()["error"]["code"] == "RUN_IN_PROGRESS"


# ── GET /eval/runs ─────────────────────────────────────────────────────────────


class TestEvalRuns:
    def test_returns_200(self, client) -> None:
        assert client.get("/eval/runs").status_code == 200

    def test_response_has_runs_and_total(self, client) -> None:
        body = client.get("/eval/runs").json()
        assert "runs" in body
        assert "total" in body
        assert isinstance(body["runs"], list)

    def test_limit_respected(self, client) -> None:
        body = client.get("/eval/runs?limit=3").json()
        assert len(body["runs"]) <= 3

    def test_limit_capped_at_100(self, client) -> None:
        body = client.get("/eval/runs?limit=999").json()
        assert len(body["runs"]) <= 100

    def test_with_synthetic_matrix_fixture(self, client, runs_dir, monkeypatch) -> None:
        """Scanning picks up the synthetic_matrix fixture and returns run summary."""
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)

        r = client.get("/eval/runs?status=all")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1

        found = next((run for run in body["runs"] if run["run_id"] == _MATRIX_ID), None)
        assert found is not None, f"Expected run_id={_MATRIX_ID} in response"
        assert "chunking" in found
        assert "embedding" in found
        assert "best_metric" in found
        assert "total_cost_usd" in found

    def test_run_summary_best_metric_shape(self, client, runs_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)

        body = client.get("/eval/runs?status=all").json()
        run = next(r for r in body["runs"] if r["run_id"] == _MATRIX_ID)
        bm = run["best_metric"]
        assert "name" in bm
        assert "value" in bm
        assert isinstance(bm["value"], float)


# ── GET /eval/runs/{run_id} ────────────────────────────────────────────────────


class TestGetRun:
    def test_unknown_run_id_returns_404(self, client) -> None:
        r = client.get("/eval/runs/nonexistent_run_xyz_999")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "RUN_NOT_FOUND"

    def test_known_matrix_id_returns_200(self, client, runs_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}")
        assert r.status_code == 200
        body = r.json()
        assert body["matrix_id"] == _MATRIX_ID

    def test_known_matrix_has_configs(self, client, runs_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)

        body = client.get(f"/eval/runs/{_MATRIX_ID}").json()
        assert "configs" in body
        assert len(body["configs"]) == 6

    def test_per_run_json_also_served(self, client, runs_dir, monkeypatch) -> None:
        """Individual EvalResult JSON files are also served by run_id."""
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)

        # The first config's run_id in the fixture
        matrix = json.loads((runs_dir / f"{_MATRIX_ID}_matrix.json").read_text())
        first_run_id = matrix["configs"][0]["run_id"]

        # Copy run file using actual run_id as filename
        src = runs_dir / matrix["configs"][0]["path"]
        dst = runs_dir / f"{first_run_id}.json"
        shutil.copy(src, dst)

        r = client.get(f"/eval/runs/{first_run_id}")
        assert r.status_code == 200


# ── GET /eval/runs/{run_id}/report ─────────────────────────────────────────────


class TestGetReport:
    def test_unknown_run_returns_404(self, client) -> None:
        r = client.get("/eval/runs/no_such_run_xyz/report")
        assert r.status_code == 404
        assert r.json()["error"]["code"] in ("RUN_NOT_FOUND", "REPORT_NOT_READY")

    def test_existing_report_returns_200_markdown(self, client, runs_dir, reports_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)
        monkeypatch.setattr(eval_mod, "_REPORTS_DIR", reports_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/report")
        assert r.status_code == 200
        assert "markdown" in r.headers.get("content-type", "")

    def test_report_content_contains_matrix_id(self, client, runs_dir, reports_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)
        monkeypatch.setattr(eval_mod, "_REPORTS_DIR", reports_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/report")
        assert _MATRIX_ID in r.text

    def test_format_html_returns_html_content_type(self, client, runs_dir, reports_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_RESULTS_DIR", runs_dir)
        monkeypatch.setattr(eval_mod, "_REPORTS_DIR", reports_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/report?format=html")
        assert r.status_code == 200
        assert "html" in r.headers.get("content-type", "")


# ── GET /eval/runs/{run_id}/figures/{name} ─────────────────────────────────────


class TestGetFigure:
    def test_unknown_figure_name_returns_404(self, client) -> None:
        r = client.get("/eval/runs/some_run/figures/unknown_name")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "FIGURE_NOT_FOUND"

    def test_missing_png_returns_404(self, client) -> None:
        r = client.get("/eval/runs/some_run/figures/heatmap")
        assert r.status_code == 404

    def test_heatmap_png_served(self, client, figures_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_FIGURES_DIR", figures_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/figures/heatmap")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_hallucination_bar_png_served(self, client, figures_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_FIGURES_DIR", figures_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/figures/hallucination_bar")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_cost_quality_scatter_png_served(self, client, figures_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_FIGURES_DIR", figures_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/figures/cost_quality_scatter")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_png_content_is_bytes(self, client, figures_dir, monkeypatch) -> None:
        import api.routes.eval as eval_mod
        monkeypatch.setattr(eval_mod, "_FIGURES_DIR", figures_dir)

        r = client.get(f"/eval/runs/{_MATRIX_ID}/figures/heatmap")
        assert len(r.content) > 0


# ── Error envelope shape ───────────────────────────────────────────────────────


class TestErrorEnvelope:
    def test_404_uses_envelope(self, client) -> None:
        body = client.get("/eval/runs/no_such_run/figures/heatmap").json()
        assert "error" in body
        for field in ("code", "message", "detail", "request_id"):
            assert field in body["error"], f"Missing error field: {field}"

    def test_400_uses_envelope(self, client) -> None:
        r = client.post("/query", json={"question": ""})
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "EMPTY_QUESTION"

    def test_request_id_in_header(self, client) -> None:
        assert "x-request-id" in client.get("/health").headers

    def test_request_id_starts_with_req_(self, client) -> None:
        rid = client.get("/health").headers.get("x-request-id", "")
        assert rid.startswith("req_")
