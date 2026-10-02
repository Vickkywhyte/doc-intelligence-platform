# CLAUDE.md — Instructions for Claude Code

> Read this file completely before making any change to this repository.
> It is the operating manual for AI-assisted work on the Document
> Intelligence Platform.

---

## 1. Project in one paragraph

A RAG system over company documents with a first-class evaluation pipeline.
Ingests PDFs, chunks 3 ways, embeds 3 ways, retrieves, generates cited
answers, and measures answer accuracy, hallucination rate, retrieval
precision, and faithfulness on a human-curated golden dataset. The
differentiator is evaluation, not the chat UI.

## 2. Required reading before any non-trivial change

1. `PROJECT_INSTRUCTIONS.md` — working principles
2. `01_PRD.md` — what we're building and why
3. `02_ARCHITECTURE.md` — how it fits together
4. `03_TECH_DECISIONS.md` — 14 ADRs (locked decisions)
5. `06_EVAL_SPEC.md` — ⭐ the differentiator; precise metric definitions
6. `04_ROADMAP.md` — the current phase's deliverables

Never contradict a locked ADR. If you believe one should change, propose an
ADR amendment and wait for approval.

## 3. Repository layout (quick reference)
doc-intelligence-platform/
├── claude_chat/ ← planning docs; do NOT modify
├── configs/ ← YAML configs
├── data/ ← corpus, chunks, indexes, golden (gitignored)
├── results/ ← run artifacts, reports, figures
├── logs/ ← structured logs + cost.jsonl
├── src/docintel/ ← library code
│ ├── ingestion/
│ ├── chunking/
│ ├── embeddings/
│ ├── vectorstore/
│ ├── retrieval/
│ ├── generation/
│ ├── pipeline/
│ ├── evaluation/
│ └── utils/
├── scripts/ ← numbered runnable entry points
├── api/ ← FastAPI
├── ui/ ← Streamlit
├── tests/ ← mirrors src/
├── notebooks/ ← exploration only
├── pyproject.toml
├── Makefile
└── docker-compose.yml


## 4. Coding rules

1. **Type hints on every function.** `from __future__ import annotations`.
2. **ABCs before implementations.** New chunker/embedder/store/metric →
   inherit from base in that package, register in the package's `registry.py`.
3. **Pydantic v2** for all data crossing module boundaries. See
   `05_DATA_MODEL.md`.
4. **No direct SDK calls** to OpenAI/Anthropic/Cohere outside
   `generation/llm_client.py` and `evaluation/judge.py`.
5. **Every LLM/embedding call emits a cost event** via
   `utils/cost_tracker.py` with `{model, prompt_tokens, completion_tokens,
   cost_usd, latency_ms}`.
6. **Config over constants.** No magic numbers. Chunk size, k, thresholds,
   model names, k_values → YAML.
7. **Determinism.** `random.seed`, `np.random.seed`, `temperature=0`,
   provider `seed=` where supported. Log any source of nondeterminism.
8. **Fail loud in dev, soft in prod.** Config/schema errors raise. Per-doc
   ingestion errors logged + skipped + summarized.
9. **No `print()`.** Use `structlog` logger.
10. **Docstrings** on every module and class.
11. **Errors:** raise typed exceptions from `docintel.exceptions`, not bare
    `Exception`.

## 5. Do NOT

- ❌ Do not install packages without updating `pyproject.toml`
- ❌ Do not call LLMs in unit tests (mock via `tests/mocks/`)
- ❌ Do not modify `data/eval/golden.jsonl` programmatically
- ❌ Do not add a metric without updating `06_EVAL_SPEC.md` + adding a test
- ❌ Do not use `print()` — use `structlog`
- ❌ Do not add files outside the current phase's deliverable list
- ❌ Do not refactor previous phases' code while in the current phase
- ❌ Do not silently change an ADR — propose an amendment
- ❌ Do not merge "hallucination" and "faithfulness" — see ADR-011, ADR-012
- ❌ Do not modify anything in `claude_chat/`

## 6. Phase discipline

Work on exactly one phase at a time, in order, from `04_ROADMAP.md`.
The current phase is stated at the top of the user's message. If unstated,
**ask before proceeding.**

Do not pre-build future phases. Do not volunteer future-phase work.

## 7. Checkpoint commands (must pass before declaring a phase done)

| Phase | Command | Expected |
|---|---|---|
| 1 | `make check-config` | Resolved config JSON printed; bad config raises |
| 2 | `make ingest` | `data/processed/*.json` present |
| 3 | `make chunk` | `data/chunks/{strategy}/*.jsonl` present |
| 4 | `make embed` | Vectors cached; 2nd run hits 100% cache |
| 5 | `make index && make query-smoke` | Top-3 chunks on-topic for seed queries |
| 6 | `make rag-smoke` | Citations present; refusal on OOD question |
| 7 | `make golden-validate` | Schema + coverage checks pass |
| 8 | `make eval-single` | `results/runs/{id}.json` produced |
| 9 | `make eval-matrix` | 9 configs, comparison report produced |
| 10 | `make serve` | FastAPI :8000 + Streamlit :8501 up |
| 11 | `make test` | All green; `docker compose up` works |

## 8. Commit message format

`[phase-N] type(scope): summary`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `eval`

Examples:
[phase-3] feat(chunking): add semantic chunker via embedding similarity
[phase-8] eval(metrics): implement claim-level hallucination rate
[phase-9] feat(report): render cost-vs-quality scatter as PNG


## 9. Response style (for AI assistant)

- **Concise and technical.** No filler.
- **Show, don't tell.** Full code blocks, exact paths, real commands.
- **Flag uncertainty explicitly.** Propose how to verify.
- **Surface trade-offs** in one sentence when a choice is non-obvious.
- **Respect "one step at a time."** Do not volunteer future phases.

## 10. When to ask vs. proceed

**Ask before proceeding if:**
- The change touches eval semantics (`06_EVAL_SPEC.md`)
- The change contradicts an ADR
- The phase is ambiguous
- A library behavior is uncertain and affects correctness
- Cost implication > $1 per run

**Proceed without asking if:**
- Clear deliverable within the current phase
- Follows an established pattern from a previous phase
- Bug fix with obvious correct behavior
- Formatting, docstrings, or type hints

## 11. When stuck

Ask. Do not guess at eval semantics. The distinction between
"hallucination" and "unfaithfulness" is intentional. The cross-strategy
gold-chunk-ID structure in `06_EVAL_SPEC.md §1.3` is intentional. The
refusal path is intentional. If any of this is unclear, stop and read the
spec again — or ask.

## 12. Session hygiene

End of session checklist:
- [ ] `CHANGELOG.md` entry added (if a phase or ADR moved)
- [ ] Phase status updated in `00_START_HERE.md`
- [ ] Any new decision captured as an ADR
- [ ] Any new metric added to `06_EVAL_SPEC.md` with test reference
- [ ] No `TODO` comments left in committed files
