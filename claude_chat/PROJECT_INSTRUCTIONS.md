# Project Instructions

> How to work in this repository. Applies to humans and AI assistants alike.

---

## 1. Mission

Build a **production-grade Document Intelligence Platform** where
**evaluation is a first-class citizen**. Every design decision must be
defensible in a technical interview or design review.

The bar: a hiring manager reading this repo should conclude
*"this person knows the difference between a RAG demo and a RAG production
system."*

## 2. Working principles

These are non-negotiable. When in doubt, refer back here.

### 2.1 Evaluate before optimizing
No tuning, no swap, no refactor of a pipeline component without a baseline
measurement. If you can't cite a metric, you can't justify the change.

### 2.2 Config over code
Chunking parameters, embedding model names, retrieval k, eval thresholds,
judge model — all YAML. Code reads config; code doesn't hardcode.

### 2.3 Interfaces over implementations
Every swappable component (chunker, embedder, vector store, retriever, LLM,
judge, metric) has an ABC. New implementations drop in without touching
call sites.

### 2.4 Reproducibility is a feature
Every experiment run produces a JSON artifact with:
- `config_hash` (SHA-256 of canonicalized YAML)
- `git_sha`
- `python_version`, `package_versions`
- `judge_model`, `judge_prompt_version`
- `seed`
- `started_at`, `ended_at`
- `total_cost_usd`

Same config + same data → same metrics (± judge noise, which is logged).

### 2.5 Cost-aware by default
Every LLM and embedding call is logged with token counts and $ estimate.
A per-run `budget_usd` hard cap aborts runaway runs. Caching is on by
default; `--no-cache` is available for benchmark purity.

### 2.6 No silent failures
- Config errors raise loudly.
- Per-document ingestion errors are logged, skipped, and summarized.
- Retrieval returning zero chunks is logged as a distinct event.
- Refusals are answers, not errors.

### 2.7 Phase discipline
Work on exactly one phase at a time from `04_ROADMAP.md`. Do not "while I'm
here, also..." Pre-building future phases creates churn and untested code.

## 3. Code conventions

### Python
- **Version:** 3.11
- **Style:** PEP 8, enforced by `ruff`
- **Types:** Required on every function; `from __future__ import annotations`
- **Docstrings:** Required on every module and class; one-line minimum
- **Imports:** Absolute (`from docintel.chunking.base import ChunkStrategy`)

### Naming
| Entity | Convention | Example |
|---|---|---|
| Module | `snake_case` | `retrieval/retriever.py` |
| Class | `PascalCase` | `RecursiveChunker` |
| Function | `snake_case` | `chunk_document()` |
| Constant | `UPPER_SNAKE` | `DEFAULT_K = 5` |
| Config key | `snake_case` | `chunk_size` |
| Test file | `test_{module}.py` | `test_chunking.py` |
| Script | `NN_verb_noun.py` | `02_chunk.py` |

### File layout
- `src/docintel/` — library code (importable package)
- `scripts/` — numbered, runnable entry points
- `configs/` — YAML only
- `tests/` — mirrors `src/` structure
- `data/`, `results/`, `logs/` — gitignored except for `.gitkeep`

## 4. Code review checklist

Self-review **before every commit**:

- [ ] Type hints on all public functions
- [ ] Docstring on every new module and class
- [ ] No magic numbers — pulled into config
- [ ] Errors logged with context, not swallowed
- [ ] Cost tracked if any LLM/embedding call made
- [ ] Corresponding test added or updated
- [ ] No `print()` — using `structlog`
- [ ] No direct SDK calls outside `llm_client.py` / `judge.py`
- [ ] Config change is backward-compatible or migration noted
- [ ] `CHANGELOG.md` updated if a decision moved

## 5. Commit discipline

**Format:** `[phase-N] type(scope): summary`

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `eval`

**Examples:**
[phase-3] feat(chunking): add semantic chunker via embedding similarity
[phase-6] fix(generation): citation parser drops page numbers >9
[phase-8] eval(metrics): tighten NDCG@5 test tolerance to 1e-6
[phase-9] docs(adr): record decision to defer cross-encoder reranker


**Before declaring a phase complete:**
1. All deliverables from `04_ROADMAP.md` present
2. Phase checkpoint command passes
3. `CHANGELOG.md` updated with a dated entry

## 6. How to work with Claude (in VSCode or chat)

### 6.1 Point Claude at the right files first
- Always: `CLAUDE.md` (operating rules)
- Before coding a module: `05_DATA_MODEL.md` + the relevant ADR
- Before touching eval: `06_EVAL_SPEC.md`

### 6.2 One phase at a time
State the phase explicitly at the top of every session:
> *"Phase 3. Build mode."*

Do not let the assistant drift into future phases. If it offers, decline.

### 6.3 Modes
- **Plan mode** — discuss, propose, no code
- **Build mode** — generate complete files for the current phase
- **Review mode** — critique existing code against ADRs and eval spec

### 6.4 When to push back
- If the assistant proposes a change contradicting an ADR → ask for an
  ADR amendment, not a silent change.
- If the assistant proposes touching eval semantics → stop, read
  `06_EVAL_SPEC.md`, then decide.
- If the assistant wants to "also refactor" something → defer to a
  dedicated cleanup phase.

## 7. Anti-patterns

Do NOT:

- ❌ Hardcode API keys, model names, or absolute paths
- ❌ Chunk without preserving `char_start` / `char_end` / `page`
- ❌ Use the same LLM as generator and judge without declaring it
- ❌ Treat unanswerable questions as failures
- ❌ Ship a metric without a unit test on a hand-crafted example
- ❌ Modify `data/eval/golden.jsonl` programmatically
- ❌ Auto-generate golden Q&A without human review
- ❌ Add a metric without updating `06_EVAL_SPEC.md`
- ❌ Install packages without updating `pyproject.toml`
- ❌ Call LLMs in unit tests (mock via `tests/mocks/`)
- ❌ Add files outside the current phase's deliverable list
- ❌ Refactor a previous phase's code while working on the current one
- ❌ Use `print()` for logging
- ❌ Mix "hallucination" and "faithfulness" (see ADR-011, ADR-012)

## 8. Testing expectations

- **Unit tests:** fast (< 1s each), no network, no LLM
- **Integration tests:** mocked LLM, full pipeline shape verification
- **Eval smoke tests:** marked `@pytest.mark.llm`, skipped by default in CI
- **Coverage:** ≥ 80% overall, ≥ 95% on `evaluation/metrics/`

Every metric in `06_EVAL_SPEC.md` has a fixture in `tests/test_metrics.py`
covering both a positive and a negative case.

## 9. Documentation discipline

- Every new decision → new ADR in `03_TECH_DECISIONS.md`
- Every new metric → new section in `06_EVAL_SPEC.md` + test reference
- Every new config key → documented in `05_DATA_MODEL.md` §config
- Every new API endpoint → added to `07_API_SPEC.md`
- Every new UI element → added to `08_UI_SPEC.md`

Documentation is not a follow-up task. It's part of "done."

## 10. Definition of "done" (per phase)

A phase is done when:

1. ✅ All deliverables listed in `04_ROADMAP.md` for that phase exist
2. ✅ Checkpoint command passes with expected output
3. ✅ New code has type hints + docstrings + tests
4. ✅ `CHANGELOG.md` has a dated entry
5. ✅ No TODO comments left in the phase's files
6. ✅ Status updated in `00_START_HERE.md`