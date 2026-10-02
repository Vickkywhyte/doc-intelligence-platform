"""Tests for the FastAPI API layer.

Uses httpx AsyncClient with the app in test mode (USE_HOSTED=false, no network).
All tests run without an OpenRouter key.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("USE_HOSTED", "false")

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    """Synchronous TestClient for all API tests (no network required)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── GET /health ────────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_200(self, client) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client) -> None:
        body = r = client.get("/health").json()
        assert body["status"] == "ok"

    def test_has_required_fields(self, client) -> None:
        body = client.get("/health").json()
        for field in ("status", "version", "git_sha", "python_version", "uptime_s"):
            assert field in body, f"Missing field: {field}"

    def test_uptime_non_negative(self, client) -> None:
        body = client.get("/health").json()
        assert body["uptime_s"] >= 0


# ── POST /query ────────────────────────────────────────────────────────────────


class TestQuery:
    def test_valid_question_returns_200(self, client) -> None:
        r = client.post("/query", json={"question": "What is in the documents?"})
        assert r.status_code == 200

    def test_response_shape(self, client) -> None:
        r = client.post("/query", json={"question": "What are the payment terms?"})
        body = r.json()
        for field in ("query", "answer", "refused", "citations", "retrieved", "model", "latency_ms", "cost_usd", "warnings"):
            assert field in body, f"Missing field: {field}"

    def test_query_echoed(self, client) -> None:
        question = "What is the notice period?"
        r = client.post("/query", json={"question": question})
        assert r.json()["query"] == question

    def test_empty_question_returns_400(self, client) -> None:
        r = client.post("/query", json={"question": "   "})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "EMPTY_QUESTION"

    def test_question_too_long_returns_400(self, client) -> None:
        r = client.post("/query", json={"question": "x" * 2001})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "QUESTION_TOO_LONG"

    def test_return_chunks_false_omits_retrieved(self, client) -> None:
        r = client.post("/query", json={"question": "test question", "return_chunks": False})
        assert r.status_code == 200
        assert r.json()["retrieved"] == []

    def test_refused_flag_is_bool(self, client) -> None:
        r = client.post("/query", json={"question": "Something totally unrelated?"})
        assert isinstance(r.json()["refused"], bool)

    def test_error_envelope_has_request_id(self, client) -> None:
        r = client.post("/query", json={"question": ""})
        body = r.json()
        assert "request_id" in body["error"]


# ── POST /ingest ───────────────────────────────────────────────────────────────


class TestIngest:
    def test_no_files_returns_400(self, client) -> None:
        r = client.post("/ingest", data={"doc_type": "contract"})
        assert r.status_code in (400, 422)

    def test_invalid_doc_type_returns_400(self, client) -> None:
        content = b"%PDF-1.4 fake pdf content"
        r = client.post(
            "/ingest",
            files=[("files", ("test.pdf", content, "application/pdf"))],
            data={"doc_type": "invalid_type"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_DOC_TYPE"

    def test_valid_pdf_returns_200_or_422(self, client) -> None:
        # A minimal valid-looking PDF; may fail OCR/parse but not schema
        content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        r = client.post(
            "/ingest",
            files=[("files", ("minimal.pdf", content, "application/pdf"))],
            data={"doc_type": "other"},
        )
        # Either 200 (processed) or 422 (all failed, which means parse error)
        assert r.status_code in (200, 422)


# ── POST /eval/run ─────────────────────────────────────────────────────────────


class TestEvalRun:
    def test_missing_config_returns_404(self, client) -> None:
        r = client.post(
            "/eval/run",
            json={"config_path": "configs/does_not_exist.yaml"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "CONFIG_NOT_FOUND"

    def test_valid_config_returns_202(self, client) -> None:
        r = client.post(
            "/eval/run",
            json={"config_path": "configs/experiment_matrix.yaml"},
        )
        # May be 202 or 409 if a run is in progress from a previous test
        assert r.status_code in (202, 409)
        if r.status_code == 202:
            body = r.json()
            assert body["status"] == "queued"
            assert "run_id" in body
            assert "n_configs" in body

    def test_queued_response_shape(self, client) -> None:
        r = client.post(
            "/eval/run",
            json={"config_path": "configs/experiment_matrix.yaml"},
        )
        if r.status_code == 202:
            body = r.json()
            for field in ("run_id", "status", "n_configs", "n_questions", "estimated_cost_usd", "estimated_duration_s"):
                assert field in body, f"Missing field: {field}"


# ── GET /eval/runs ─────────────────────────────────────────────────────────────


class TestEvalRuns:
    def test_returns_200(self, client) -> None:
        r = client.get("/eval/runs")
        assert r.status_code == 200

    def test_response_shape(self, client) -> None:
        body = client.get("/eval/runs").json()
        assert "runs" in body
        assert "total" in body
        assert isinstance(body["runs"], list)

    def test_limit_param(self, client) -> None:
        r = client.get("/eval/runs?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert len(body["runs"]) <= 5

    def test_with_synthetic_matrix(self, client, tmp_path, monkeypatch) -> None:
        """Verify scanning picks up a *_matrix.json file."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        matrix_data = {
            "matrix_id": "test_matrix_123",
            "configs": [
                {
                    "config_code": "fix·bge",
                    "run_id": "run_001",
                    "path": "run_001.json",
                    "aggregate_metrics": {"answer_accuracy": 0.8},
                }
            ],
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T01:00:00Z",
            "total_cost_usd": 0.1,
            "n_configs_completed": 1,
            "n_configs_total": 1,
            "status": "complete",
        }
        (runs_dir / "test_matrix_123_matrix.json").write_text(json.dumps(matrix_data))
        import api.routes.eval as eval_module
        monkeypatch.setattr(eval_module, "_RESULTS_DIR", runs_dir)
        r = client.get("/eval/runs?status=all")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        assert body["runs"][0]["run_id"] == "test_matrix_123"


# ── GET /eval/runs/{run_id} ────────────────────────────────────────────────────


class TestGetRun:
    def test_unknown_run_returns_404(self, client) -> None:
        r = client.get("/eval/runs/nonexistent_run_id_xyz")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "RUN_NOT_FOUND"

    def test_found_run_returns_json(self, client, tmp_path, monkeypatch) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        data = {"run_id": "test_run_abc", "n_questions": 10}
        (runs_dir / "test_run_abc.json").write_text(json.dumps(data))
        import api.routes.eval as eval_module
        monkeypatch.setattr(eval_module, "_RESULTS_DIR", runs_dir)
        r = client.get("/eval/runs/test_run_abc")
        assert r.status_code == 200
        assert r.json()["run_id"] == "test_run_abc"


# ── GET /eval/runs/{run_id}/report ─────────────────────────────────────────────


class TestGetReport:
    def test_missing_run_returns_404(self, client) -> None:
        r = client.get("/eval/runs/nonexistent_run_xyz/report")
        assert r.status_code == 404

    def test_error_code_run_not_found(self, client) -> None:
        r = client.get("/eval/runs/nonexistent_run_xyz/report")
        assert r.json()["error"]["code"] in ("RUN_NOT_FOUND", "REPORT_NOT_READY")


# ── GET /eval/runs/{run_id}/figures/{name} ─────────────────────────────────────


class TestGetFigure:
    def test_unknown_figure_name_returns_404(self, client) -> None:
        r = client.get("/eval/runs/some_run/figures/unknown_figure")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "FIGURE_NOT_FOUND"

    def test_missing_png_returns_404(self, client) -> None:
        r = client.get("/eval/runs/some_run/figures/heatmap")
        assert r.status_code == 404

    def test_existing_png_served(self, client, tmp_path, monkeypatch) -> None:
        figs_dir = tmp_path / "figures"
        figs_dir.mkdir()
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        (figs_dir / "my_run_heatmap.png").write_bytes(png_content)
        import api.routes.eval as eval_module
        monkeypatch.setattr(eval_module, "_FIGURES_DIR", figs_dir)
        r = client.get("/eval/runs/my_run/figures/heatmap")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"


# ── Error envelope shape ───────────────────────────────────────────────────────


class TestErrorEnvelope:
    def test_404_uses_error_envelope(self, client) -> None:
        r = client.get("/eval/runs/no_such_run/figures/heatmap")
        body = r.json()
        assert "error" in body
        error = body["error"]
        for field in ("code", "message", "detail", "request_id"):
            assert field in error, f"Missing error field: {field}"

    def test_request_id_header_set(self, client) -> None:
        r = client.get("/health")
        assert "x-request-id" in r.headers
