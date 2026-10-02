"""Phase 7 — Schema validation tests for golden dataset models and golden.jsonl.

Tests:
  - GoldenItem Pydantic model (happy path, error cases)
  - golden.jsonl file composition and chunk ID integrity
  - 06_validate_golden.py exit code
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.evaluation.models import GoldenItem  # noqa: E402

GOLDEN_PATH = Path("data/eval/golden.jsonl")
CHUNKS_DIR = Path("data/chunks")
STRATEGIES = ("fixed", "recursive", "semantic")

REQUIRED_FACTUAL = 45
REQUIRED_MULTI_HOP = 15
REQUIRED_UNANSWERABLE = 15
TOTAL = REQUIRED_FACTUAL + REQUIRED_MULTI_HOP + REQUIRED_UNANSWERABLE


# ── GoldenItem model unit tests ───────────────────────────────────────────────


class TestGoldenItemModel:
    def _factual(self, **kwargs) -> dict:
        base = {
            "id": "g001",
            "question": "What are the payment terms?",
            "gold_answer": "Payment is due within 30 days of invoice.",
            "gold_chunk_ids": {
                "fixed": ["abc123:fixed:00000"],
                "recursive": ["abc123:recursive:00000"],
                "semantic": ["abc123:semantic:00000"],
            },
            "type": "factual",
            "notes": "test",
        }
        base.update(kwargs)
        return base

    def test_valid_factual(self):
        item = GoldenItem.model_validate(self._factual())
        assert item.type == "factual"
        assert item.id == "g001"

    def test_valid_multi_hop(self):
        item = GoldenItem.model_validate(self._factual(type="multi_hop"))
        assert item.type == "multi_hop"

    def test_valid_unanswerable(self):
        item = GoldenItem.model_validate(
            {
                "id": "g060",
                "question": "What is the annual bonus?",
                "gold_answer": "This information is not available in the provided documents.",
                "gold_chunk_ids": {},
                "type": "unanswerable",
                "notes": "",
            }
        )
        assert item.gold_chunk_ids == {}

    def test_missing_id_raises(self):
        raw = self._factual()
        del raw["id"]
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_missing_question_raises(self):
        raw = self._factual()
        del raw["question"]
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_missing_gold_answer_raises(self):
        raw = self._factual()
        del raw["gold_answer"]
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_missing_type_raises(self):
        raw = self._factual()
        del raw["type"]
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_invalid_type_raises(self):
        raw = self._factual(type="hallucinated")
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_extra_fields_forbidden(self):
        raw = self._factual()
        raw["unexpected_field"] = "oops"
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_notes_default_empty_string(self):
        raw = self._factual()
        del raw["notes"]
        item = GoldenItem.model_validate(raw)
        assert item.notes == ""

    def test_gold_chunk_ids_empty_allowed_for_unanswerable(self):
        item = GoldenItem.model_validate(
            {
                "id": "g075",
                "question": "What is the overtime rate?",
                "gold_answer": "This information is not available in the provided documents.",
                "gold_chunk_ids": {},
                "type": "unanswerable",
            }
        )
        assert item.gold_chunk_ids == {}

    def test_multiple_chunk_ids_per_strategy(self):
        raw = self._factual(
            gold_chunk_ids={
                "fixed": ["abc:fixed:00000", "abc:fixed:00001"],
                "recursive": ["abc:recursive:00000"],
                "semantic": ["abc:semantic:00000"],
            }
        )
        item = GoldenItem.model_validate(raw)
        assert len(item.gold_chunk_ids["fixed"]) == 2

    def test_id_must_not_be_empty(self):
        raw = self._factual(id="")
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_question_must_be_at_least_5_chars(self):
        raw = self._factual(question="Why?")
        with pytest.raises(ValidationError):
            GoldenItem.model_validate(raw)

    def test_serialization_roundtrip(self):
        raw = self._factual()
        item = GoldenItem.model_validate(raw)
        dumped = item.model_dump()
        item2 = GoldenItem.model_validate(dumped)
        assert item == item2


# ── golden.jsonl file tests ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def golden_items() -> list[GoldenItem]:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"{GOLDEN_PATH} not found — run `make golden-seed` first")
    lines = GOLDEN_PATH.read_text().splitlines()
    return [GoldenItem.model_validate(json.loads(l)) for l in lines if l.strip()]


@pytest.fixture(scope="module")
def known_chunk_ids() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for strategy in STRATEGIES:
        result[strategy] = set()
        for jsonl in sorted((CHUNKS_DIR / strategy).glob("*.jsonl")):
            for line in jsonl.read_text().splitlines():
                if line.strip():
                    c = json.loads(line)
                    result[strategy].add(c["chunk_id"])
    return result


class TestGoldenFile:
    def test_file_exists(self):
        assert GOLDEN_PATH.exists(), f"golden.jsonl not found at {GOLDEN_PATH}"

    def test_total_count(self, golden_items: list[GoldenItem]):
        assert len(golden_items) == TOTAL, f"Expected {TOTAL}, got {len(golden_items)}"

    def test_factual_count(self, golden_items: list[GoldenItem]):
        n = sum(1 for g in golden_items if g.type == "factual")
        assert n == REQUIRED_FACTUAL, f"Expected {REQUIRED_FACTUAL} factual, got {n}"

    def test_multi_hop_count(self, golden_items: list[GoldenItem]):
        n = sum(1 for g in golden_items if g.type == "multi_hop")
        assert n == REQUIRED_MULTI_HOP, f"Expected {REQUIRED_MULTI_HOP} multi_hop, got {n}"

    def test_unanswerable_count(self, golden_items: list[GoldenItem]):
        n = sum(1 for g in golden_items if g.type == "unanswerable")
        assert n == REQUIRED_UNANSWERABLE, f"Expected {REQUIRED_UNANSWERABLE} unanswerable, got {n}"

    def test_unique_ids(self, golden_items: list[GoldenItem]):
        ids = [g.id for g in golden_items]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in golden.jsonl"

    def test_unanswerable_have_empty_chunk_ids(self, golden_items: list[GoldenItem]):
        for item in golden_items:
            if item.type == "unanswerable":
                assert item.gold_chunk_ids == {}, (
                    f"[{item.id}] unanswerable entry has non-empty gold_chunk_ids"
                )

    def test_factual_and_multihop_have_all_strategies(self, golden_items: list[GoldenItem]):
        for item in golden_items:
            if item.type in ("factual", "multi_hop"):
                missing = [s for s in STRATEGIES if s not in item.gold_chunk_ids]
                assert not missing, (
                    f"[{item.id}] missing strategies: {missing}"
                )

    def test_factual_and_multihop_chunk_ids_not_empty(self, golden_items: list[GoldenItem]):
        for item in golden_items:
            if item.type in ("factual", "multi_hop"):
                for strategy, ids in item.gold_chunk_ids.items():
                    assert len(ids) > 0, (
                        f"[{item.id}] strategy '{strategy}' has empty chunk ID list"
                    )

    def test_all_chunk_ids_exist(
        self,
        golden_items: list[GoldenItem],
        known_chunk_ids: dict[str, set[str]],
    ):
        bad: list[str] = []
        for item in golden_items:
            for strategy, ids in item.gold_chunk_ids.items():
                if strategy not in known_chunk_ids:
                    bad.append(f"[{item.id}] unknown strategy '{strategy}'")
                    continue
                for cid in ids:
                    if cid not in known_chunk_ids[strategy]:
                        bad.append(f"[{item.id}] chunk_id '{cid}' not in {strategy} index")
        assert not bad, "Chunk ID integrity failures:\n" + "\n".join(bad)

    def test_questions_not_empty(self, golden_items: list[GoldenItem]):
        for item in golden_items:
            assert len(item.question.strip()) >= 5, f"[{item.id}] question too short"

    def test_gold_answers_not_empty(self, golden_items: list[GoldenItem]):
        for item in golden_items:
            assert len(item.gold_answer.strip()) > 0, f"[{item.id}] empty gold_answer"

    def test_all_entries_parse_as_valid_json(self):
        lines = GOLDEN_PATH.read_text().splitlines()
        for i, line in enumerate(lines, start=1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    pytest.fail(f"Line {i}: invalid JSON — {exc}")

    def test_multiple_doc_types_covered(self, golden_items: list[GoldenItem]):
        """Verify diverse corpus coverage: at least 8 distinct doc IDs referenced."""
        doc_ids: set[str] = set()
        for item in golden_items:
            for ids in item.gold_chunk_ids.values():
                for cid in ids:
                    doc_ids.add(cid.split(":")[0])
        assert len(doc_ids) >= 8, (
            f"Expected at least 8 distinct doc_ids, got {len(doc_ids)}: {doc_ids}"
        )


# ── Validator script exit code ────────────────────────────────────────────────


class TestValidatorScript:
    def test_validate_golden_exits_zero(self):
        if not GOLDEN_PATH.exists():
            pytest.skip(f"{GOLDEN_PATH} not found")
        result = subprocess.run(
            [sys.executable, "scripts/06_validate_golden.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"06_validate_golden.py exited {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_validate_nonexistent_file_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "scripts/06_validate_golden.py", str(tmp_path / "missing.jsonl")],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
