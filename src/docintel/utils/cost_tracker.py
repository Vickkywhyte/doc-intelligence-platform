"""Append-only cost ledger and budget guard.

One global CostTracker instance is created at import time. All LLM, embedding,
and judge calls must emit a cost event via track():

    from docintel.utils.cost_tracker import cost_tracker
    cost_tracker.track(run_id="...", kind="llm", model="openai/gpt-4o-mini",
                       prompt_tokens=842, completion_tokens=67, cost_usd=0.00021)

The ledger is flushed to logs/cost.jsonl as append-only JSONL.
Budget enforcement is per-run: call reset_run(budget_usd) at run start.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import structlog

from docintel.exceptions import BudgetExceededError

log = structlog.get_logger()

CostKind = Literal["llm", "embedding", "judge"]

_WARN_THRESHOLD = 0.80  # emit warning at 80% of budget


class CostTracker:
    """Thread-safe, append-only cost ledger with per-run budget enforcement."""

    def __init__(self, ledger_path: str | Path = "logs/cost.jsonl") -> None:
        self._ledger_path = Path(ledger_path)
        self._lock = threading.Lock()
        self._session_total: float = 0.0
        self._run_total: float = 0.0
        self._run_budget: float | None = None
        self._run_id: str | None = None
        self._warned: bool = False

    def reset_run(self, run_id: str, budget_usd: float) -> None:
        """Call at the start of an eval run to reset per-run counters."""
        with self._lock:
            self._run_id = run_id
            self._run_budget = budget_usd
            self._run_total = 0.0
            self._warned = False

    @property
    def session_total(self) -> float:
        return self._session_total

    @property
    def run_total(self) -> float:
        return self._run_total

    def track(
        self,
        *,
        kind: CostKind,
        model: str,
        cost_usd: float,
        run_id: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        tokens: int | None = None,
    ) -> None:
        """Record one billable call and check budget.

        Raises BudgetExceededError if the run budget is exceeded.
        """
        effective_run_id = run_id or self._run_id

        event: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": effective_run_id,
            "kind": kind,
            "model": model,
            "cost_usd": cost_usd,
        }
        if prompt_tokens is not None:
            event["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            event["completion_tokens"] = completion_tokens
        if tokens is not None:
            event["tokens"] = tokens

        with self._lock:
            self._session_total += cost_usd
            self._run_total += cost_usd
            self._flush(event)
            self._check_budget()

    def _flush(self, event: dict) -> None:
        """Append a single JSON line to the ledger (called under lock)."""
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def _check_budget(self) -> None:
        """Emit warnings and raise if budget is exceeded (called under lock)."""
        if self._run_budget is None:
            return

        fraction = self._run_total / self._run_budget if self._run_budget > 0 else 0.0

        if fraction >= 1.0:
            log.error(
                "cost_budget_exceeded",
                run_id=self._run_id,
                spent=self._run_total,
                budget=self._run_budget,
            )
            raise BudgetExceededError(self._run_total, self._run_budget)

        if fraction >= _WARN_THRESHOLD and not self._warned:
            self._warned = True
            log.warning(
                "cost_budget_warning",
                run_id=self._run_id,
                spent=self._run_total,
                budget=self._run_budget,
                pct=round(fraction * 100, 1),
            )


# Module-level singleton — import and use directly.
cost_tracker = CostTracker()
