"""Eval routes: POST /eval/run, GET /eval/runs, GET /eval/runs/{id}, report, figures."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response

from api.schemas import (
    BestMetric,
    EvalRunQueued,
    EvalRunRequest,
    RunSummary,
    RunsListResponse,
)

log = structlog.get_logger()
router = APIRouter()

_RESULTS_DIR = Path("results/runs")
_REPORTS_DIR = Path("results/reports")
_FIGURES_DIR = Path("results/figures")
_CANONICAL_CONFIG = "configs/experiment_matrix.yaml"

# Global run state — only one matrix run at a time
_run_lock = threading.Lock()
_active_run_id: str | None = None


# ── POST /eval/run ─────────────────────────────────────────────────────────────


@router.post("/eval/run", status_code=202, response_model=EvalRunQueued)
async def start_eval_run(body: EvalRunRequest, request: Request) -> EvalRunQueued:
    """Enqueue a full matrix run (asynchronous)."""
    global _active_run_id
    request_id = getattr(request.state, "request_id", "")

    config_path = Path(body.config_path)
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "CONFIG_NOT_FOUND",
                "message": f"Config file not found: {body.config_path}",
                "detail": {"config_path": body.config_path},
                "request_id": request_id,
            },
        )

    if not _run_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUN_IN_PROGRESS",
                "message": "A matrix run is already in progress.",
                "detail": {"active_run_id": _active_run_id},
                "request_id": request_id,
            },
        )

    try:
        from docintel.utils.config import load_config
        cfg = load_config(str(config_path))
    except Exception as exc:
        _run_lock.release()
        raise HTTPException(
            status_code=404,
            detail={
                "code": "CONFIG_NOT_FOUND",
                "message": str(exc),
                "detail": {},
                "request_id": request_id,
            },
        )

    n_configs = len(cfg.chunking) * len(cfg.embeddings)
    n_questions = _estimate_n_questions()
    # Rough cost estimate: ~$0.002 per question per config
    estimated_cost = n_configs * n_questions * 0.002
    estimated_duration = n_configs * n_questions * 0.3  # ~0.3s per question

    import datetime
    import uuid
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    _active_run_id = run_id

    thread = threading.Thread(
        target=_run_matrix_background,
        args=(run_id, str(config_path), body.notes),
        daemon=True,
    )
    thread.start()

    log.info("eval_run_queued", run_id=run_id, n_configs=n_configs)

    return EvalRunQueued(
        run_id=run_id,
        status="queued",
        n_configs=n_configs,
        n_questions=n_questions,
        estimated_cost_usd=round(estimated_cost, 4),
        estimated_duration_s=round(estimated_duration, 1),
    )


def _estimate_n_questions() -> int:
    """Return the number of golden questions or a default estimate."""
    golden = Path("data/eval/golden.jsonl")
    if not golden.exists():
        return 75
    try:
        return sum(1 for line in golden.read_text().splitlines() if line.strip())
    except Exception:
        return 75


def _run_matrix_background(run_id: str, config_path: str, notes: str | None) -> None:
    """Run the matrix in a background thread; release lock when done."""
    global _active_run_id
    try:
        from docintel.pipeline.experiment import run_matrix
        run_matrix(config_path=config_path, output_dir="results/")
        log.info("eval_run_complete_background", run_id=run_id)
    except Exception as exc:
        log.error("eval_run_failed_background", run_id=run_id, error=str(exc))
    finally:
        _active_run_id = None
        try:
            _run_lock.release()
        except RuntimeError:
            pass


# ── GET /eval/runs ─────────────────────────────────────────────────────────────


@router.get("/eval/runs", response_model=RunsListResponse)
async def list_runs(
    limit: int = 20,
    status: str = "completed",
) -> RunsListResponse:
    """List past runs newest-first, scanning *_matrix.json files."""
    limit = min(limit, 100)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix_files = sorted(_RESULTS_DIR.glob("*_matrix.json"), reverse=True)
    summaries: list[RunSummary] = []

    for path in matrix_files:
        try:
            data = json.loads(path.read_text())
            run_status = _map_status(data.get("status", "complete"))
            if status != "all" and run_status != status:
                continue

            configs = data.get("configs", [])
            first_cfg = configs[0] if configs else {}
            first_code = first_cfg.get("config_code", "")
            chunking, embedding = _parse_config_code(first_code)

            # Pick best accuracy metric
            agg = first_cfg.get("aggregate_metrics", {})
            best_val = agg.get("answer_accuracy", 0.0)
            best_metric = BestMetric(name="answer_accuracy", value=round(best_val, 4))

            # Load first child run for n_questions + config_hash
            first_run_path = _RESULTS_DIR / first_cfg.get("path", "")
            n_questions = 0
            config_hash = ""
            if first_run_path.exists():
                try:
                    child = json.loads(first_run_path.read_text())
                    n_questions = child.get("n_questions", 0)
                    config_hash = child.get("config_hash", "")
                except Exception:
                    pass

            summaries.append(
                RunSummary(
                    run_id=data.get("matrix_id", path.stem),
                    config_hash=config_hash,
                    chunking=chunking,
                    embedding=embedding,
                    started_at=data.get("started_at", ""),
                    ended_at=data.get("ended_at", ""),
                    n_questions=n_questions,
                    best_metric=best_metric,
                    total_cost_usd=data.get("total_cost_usd", 0.0),
                    status=run_status,
                )
            )
        except Exception as exc:
            log.warning("runs_list_parse_error", path=str(path), error=str(exc))
            continue

        if len(summaries) >= limit:
            break

    return RunsListResponse(runs=summaries, total=len(summaries))


def _map_status(raw: str) -> str:
    if raw == "complete":
        return "completed"
    if raw == "aborted":
        return "failed"
    return raw


def _parse_config_code(code: str) -> tuple[str, str]:
    """Parse 'fix·bge' → ('fix', 'bge')."""
    parts = code.split("·")
    if len(parts) == 2:
        return parts[0], parts[1]
    return code, ""


# ── GET /eval/runs/{run_id} ────────────────────────────────────────────────────


@router.get("/eval/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> Response:
    """Return the full EvalResult or MatrixResult JSON for a run."""
    request_id = getattr(request.state, "request_id", "")

    # Try matrix file first, then individual EvalResult
    matrix_path = _RESULTS_DIR / f"{run_id}_matrix.json"
    direct_path = _RESULTS_DIR / f"{run_id}.json"

    for path in (matrix_path, direct_path):
        if path.exists():
            return Response(content=path.read_text(), media_type="application/json")

    raise HTTPException(
        status_code=404,
        detail={
            "code": "RUN_NOT_FOUND",
            "message": f"Run not found: {run_id}",
            "detail": {"run_id": run_id},
            "request_id": request_id,
        },
    )


# ── GET /eval/runs/{run_id}/report ─────────────────────────────────────────────


@router.get("/eval/runs/{run_id}/report")
async def get_report(run_id: str, request: Request, format: str = "md") -> Response:
    """Return the Markdown or HTML report for a matrix run."""
    request_id = getattr(request.state, "request_id", "")

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = _REPORTS_DIR / f"{run_id}.md"

    # Check that the run exists at all
    matrix_path = _RESULTS_DIR / f"{run_id}_matrix.json"
    if not matrix_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "RUN_NOT_FOUND",
                "message": f"Run not found: {run_id}",
                "detail": {"run_id": run_id},
                "request_id": request_id,
            },
        )

    if not md_path.exists():
        # Try to generate it on-demand
        try:
            from docintel.evaluation.report import generate_report
            generate_report(matrix_path=str(matrix_path), output_dir="results/")
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "REPORT_NOT_READY",
                    "message": f"Report not yet generated: {exc}",
                    "detail": {"run_id": run_id},
                    "request_id": request_id,
                },
            )

    content = md_path.read_text(encoding="utf-8")

    if format == "html":
        try:
            import markdown
            html = markdown.markdown(content, extensions=["tables"])
        except ImportError:
            html = f"<pre>{content}</pre>"
        return Response(content=html, media_type="text/html")

    return PlainTextResponse(content=content, media_type="text/markdown")


# ── GET /eval/runs/{run_id}/figures/{name} ─────────────────────────────────────

_FIGURE_SUFFIX: dict[str, str] = {
    "heatmap": "heatmap",
    "hallucination_bar": "hallucination_bar",
    "cost_quality_scatter": "cost_quality_scatter",
}


@router.get("/eval/runs/{run_id}/figures/{name}")
async def get_figure(run_id: str, name: str, request: Request) -> FileResponse:
    """Serve a generated PNG figure."""
    request_id = getattr(request.state, "request_id", "")

    if name not in _FIGURE_SUFFIX:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "FIGURE_NOT_FOUND",
                "message": f"Unknown figure name: {name}. Valid names: {list(_FIGURE_SUFFIX)}",
                "detail": {"name": name},
                "request_id": request_id,
            },
        )

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = _FIGURES_DIR / f"{run_id}_{_FIGURE_SUFFIX[name]}.png"

    if not png_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "FIGURE_NOT_FOUND",
                "message": f"Figure not found: {png_path.name}",
                "detail": {"run_id": run_id, "name": name},
                "request_id": request_id,
            },
        )

    return FileResponse(path=str(png_path), media_type="image/png")
