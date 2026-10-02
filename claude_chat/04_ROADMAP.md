# Roadmap

**Related docs:** `01_PRD.md`, `02_ARCHITECTURE.md`, `05_DATA_MODEL.md`

---

## How to use this document

- **Work one phase at a time, in order.**
- Do not start Phase N+1 until Phase N's checkpoint command passes.
- A phase is done only when all deliverables exist and the checkpoint passes.
- Update `00_START_HERE.md` status table and `CHANGELOG.md` when a phase closes.

**Legend:** 🟢 done · 🟡 in progress · ⚪ not started · 🔴 blocked

---

## Phase overview

| # | Phase | Est. effort | Depends on |
|---|---|---|---|
| 1 | Scaffolding & Config | 0.5 day | — |
| 2 | Ingestion | 1 day | 1 |
| 3 | Chunking | 1 day | 2 |
| 4 | Embeddings | 0.5 day | 3 |
| 5 | Vector Store & Retrieval | 0.5 day | 4 |
| 6 | Generation | 1 day | 5 |
| 7 | Golden Dataset | 1 day | 6 |
| 8 | Evaluation Metrics | 2 days | 7 |
| 9 | Experiment Matrix & Reporting | 1 day | 8 |
| 10 | API & UI | 1.5 days | 9 |
| 11 | Hardening (tests, Docker, docs) | 1 day | 10 |
| | **Total** | **~11 days** | |

---

## Phase 1 — Scaffolding & Config 🟢

**Goal:** A cloneable repo skeleton with dependency management, config loader,
and a green test run.

### Deliverables
```
pyproject.toml
.env.example
.gitignore
Makefile
docker-compose.yml                 # stub; finalized in Phase 11
src/docintel/__init__.py
src/docintel/utils/__init__.py
src/docintel/utils/config.py       # YAML → Pydantic, with hash()
src/docintel/utils/logging.py      # structlog setup
src/docintel/utils/cost_tracker.py # skeleton with append-only JSONL
src/docintel/exceptions.py         # typed exceptions
configs/experiment_matrix.yaml     # canonical config
configs/prices.yaml                # token pricing table
tests/__init__.py
tests/test_config.py
```

### Acceptance criteria
- `pip install -e ".[dev]"` succeeds on a clean Python 3.11 env.
- `python -c "from docintel.utils.config import load_config"` imports cleanly.
- Loading `configs/experiment_matrix.yaml` returns a Pydantic object.
- A deliberately malformed config raises a typed `ConfigError`.
- `config_hash()` is stable: same YAML → same hash across processes.
- `pytest -q` passes (at least 3 tests).

### Checkpoint
```bash
make check-config
```
**Expected:** Prints resolved config as JSON, then `config_hash=<hex>` and
exits 0. Running with `CONFIG=configs/broken.yaml` exits 1 with a
`ConfigError` message.

### Risks
- Pydantic v2 strict mode surprise: use `extra="forbid"` — catch typos early.

---

## Phase 2 — Ingestion 🟢

**Goal:** PDFs/DOCX/TXT → page-aware `Document` objects on disk.

### Deliverables
```
src/docintel/ingestion/__init__.py
src/docintel/ingestion/loader.py       # Unstructured + pdfplumber fallback
src/docintel/ingestion/cleaner.py      # whitespace, dehyphenation, header/footer
src/docintel/ingestion/metadata.py     # doc_type inference, mtime, size
scripts/00_synthesize_corpus.py        # generates ~12 realistic PDFs
scripts/01_ingest.py
tests/test_ingestion.py
```

### Acceptance criteria
- `scripts/00_synthesize_corpus.py` writes ~12 PDFs to `data/raw/`
  (MSAs, NDAs, policies, handbooks, DPAs) with realistic layouts.
- `scripts/01_ingest.py` writes `data/processed/{doc_id}.json` for each doc.
- Round-trip check: concatenating `pages[].text` reproduces cleaned source
  within whitespace tolerance.
- `doc_id = sha256(file_bytes)[:16]` — stable across machines.
- Ingestion summary printed: `processed=N, failed=M, total_chars=...`.
- A corrupted PDF is skipped with a logged error, not a crash.

### Checkpoint
```bash
make corpus ingest
```
**Expected:** `data/raw/*.pdf` populated; `data/processed/*.json` populated;
summary line shows `processed ≥ 10`, `failed = 0`.

### Risks
- `unstructured` needs `poppler` and `tesseract` system deps. Document in
  README; provide `pdfplumber` fallback that works without them.

---

## Phase 3 — Chunking 🟢

**Goal:** Three chunking strategies behind one ABC, all runnable from config.

### Deliverables
```
src/docintel/chunking/__init__.py
src/docintel/chunking/base.py          # ChunkStrategy ABC
src/docintel/chunking/fixed.py         # token-aware fixed windows
src/docintel/chunking/recursive.py     # separator hierarchy
src/docintel/chunking/semantic.py      # embedding-similarity valleys
src/docintel/chunking/registry.py      # name → class
scripts/02_chunk.py
tests/test_chunking.py
tests/fixtures/sample_document.json
```

### Acceptance criteria
- Each strategy returns `list[Chunk]` where every chunk has valid
  `char_start`, `char_end`, `page`, `token_count`, and a unique `chunk_id`.
- Every chunk's `char_start`/`char_end` maps into the source doc's
  concatenated text.
- Overlap is honored: `overlap_tokens` from config.
- `chunk_id = f"{doc_id}:{strategy}:{index:05d}"`.
- No chunk exceeds `max_size` (semantic) or `size + overlap` (fixed/recursive).
- Running on all docs produces `data/chunks/{strategy}/*.jsonl`.

### Checkpoint
```bash
make chunk
```
**Expected:** `data/chunks/fixed/`, `data/chunks/recursive/`,
`data/chunks/semantic/` each contain one `.jsonl` per doc, with summary
line per strategy: `strategy=X docs=N chunks=M avg_tokens=...`.

### Risks
- Semantic chunker needs embeddings *before* chunking → chicken-and-egg.
  **Resolution:** use the cheapest embedder (MiniLM) for the semantic splitter
  only. Documented in ADR-004.

---

## Phase 4 — Embeddings 🟢

**Goal:** Three embedders behind one ABC, with content-hash caching.

### Deliverables
```
src/docintel/embeddings/__init__.py
src/docintel/embeddings/base.py             # Embedder ABC
src/docintel/embeddings/openai_embedder.py  # via OpenRouter (ADR-015, new)
src/docintel/embeddings/bge_embedder.py
src/docintel/embeddings/minilm_embedder.py
src/docintel/embeddings/registry.py
src/docintel/utils/cache.py                 # hash-keyed .npy store
tests/test_embeddings.py                    # mocked; no network
```

### Acceptance criteria
- `Embedder.embed(texts) -> np.ndarray[N, dim]` with `dim` matching the model.
- Batching honored: `batch_size` from config.
- Cache hit on identical `(text, model_name)` returns same vector without
  re-computation.
- Second full pass over the corpus reports `cache_hit_rate=1.0` and
  `cost_usd=0.00` for hosted embedder.
- Local models download on first use and cache in `~/.cache/huggingface/`.

### Checkpoint
```bash
make embed
```
**Expected:** First run: `cache_hits=0`. Second run: `cache_hits=N`,
`cache_hit_rate=1.0`, `cost_usd=0.00`.

### Risks
- Local model download ≈ 90 MB (BGE) + 90 MB (MiniLM). Pre-cache step in
  `make install` optional. Document in README.
- **OpenRouter:** embeddings via OpenRouter use the OpenAI-compatible
  `/embeddings` endpoint. Model name becomes
  `openai/text-embedding-3-small`. See ADR-015 (added when Phase 4 begins).

---

## Phase 5 — Vector Store & Retrieval 🟢

**Goal:** Persist vectors in Chroma; retrieve top-k (with optional MMR).

### Deliverables
```
src/docintel/vectorstore/__init__.py
src/docintel/vectorstore/base.py           # VectorStore ABC
src/docintel/vectorstore/chroma_store.py
src/docintel/vectorstore/registry.py
src/docintel/retrieval/__init__.py
src/docintel/retrieval/retriever.py        # top-k + MMR
src/docintel/retrieval/reranker.py         # no-op stub for v1.1
scripts/03_embed_and_index.py
scripts/03b_query_smoke.py
tests/test_retrieval.py
```

### Acceptance criteria
- `VectorStore.upsert(chunks, vectors)` writes to
  `data/indexes/{strategy}_{embedder}/`.
- `VectorStore.query(vector, k, where=...)` returns ranked `RetrievedChunk`s
  with score + rank.
- MMR retrieval works and de-duplicates near-identical chunks.
- Metadata filter (`doc_type=contract`) narrows results.
- `query-smoke` returns on-topic chunks for 3 hand-picked questions.

### Checkpoint
```bash
make index query-smoke
```
**Expected:** 9 index directories under `data/indexes/`. Smoke script prints
top-3 chunks for 3 seed questions; a human eyeball confirms they're on-topic.

### Risks
- Chroma's default distance is L2. We use cosine — must set
  `collection_metadata={"hnsw:space": "cosine"}` explicitly. Documented.

---

## Phase 6 — Generation 🟢

**Goal:** Grounded answers with citations; explicit refusal path.

### Deliverables
```
src/docintel/generation/__init__.py
src/docintel/generation/llm_client.py      # OpenRouter + Anthropic behind ABC
src/docintel/generation/prompt_templates.py
src/docintel/generation/citation_parser.py # regex → Citation[]
src/docintel/pipeline/__init__.py
src/docintel/pipeline/rag_pipeline.py
scripts/04_rag_smoke.py
tests/test_prompt.py
tests/test_citation_parser.py
```

### Acceptance criteria
- `RAGPipeline.query(q) -> RAGResponse` with answer, citations, retrieved,
  refused, latency_ms, cost_usd.
- Prompt template enforces: (a) use only provided context, (b) cite sources
  as `[doc:page:chunk]`, (c) refuse if context insufficient.
- Citation parser extracts all `[...]` citations; malformed ones are logged,
  not silently dropped.
- Refusal detection: explicit phrase list + LLM-judge confirmation for the
  ambiguous cases (in tests only; in prod the phrase list is used).
- Smoke: 3 in-corpus questions answer correctly with citations; 1 OOD
  question returns `refused=True`.

### Checkpoint
```bash
make rag-smoke
```
**Expected:** 3 answers with ≥ 1 citation each; 1 refusal with
`refused=True`. Latency + cost printed per query.

### Risks
- LLM may hallucinate a citation format that doesn't exist. Parser must
  validate against retrieved chunk IDs and log mismatches.
- **OpenRouter:** generation goes through `openrouter.ai/api/v1/chat/completions`.
  Model name becomes `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`, etc.
  We use OpenRouter's `/models` endpoint to validate model availability at
  startup and fail fast if a configured model is deprecated.

---

## Phase 7 — Golden Dataset 🟢

**Goal:** 75 human-verified Q&A pairs with cross-strategy gold chunk IDs.

### Deliverables
```
scripts/05_seed_golden.py             # synthetic seed (reviewed, not auto-committed)
scripts/06_validate_golden.py
data/eval/golden.jsonl                # the artifact (human-curated)
data/eval/README.md                   # curation rules and coverage report
tests/test_golden_schema.py
```

### Acceptance criteria
- 75 entries: 45 factual · 15 multi_hop · 15 unanswerable.
- Every factual/multi_hop entry has `gold_chunk_ids` for all 3 strategies.
- Every `gold_chunk_id` exists in `data/chunks/{strategy}/`.
- Every `gold_answer` ≤ 2 sentences.
- Unanswerable entries have `gold_chunk_ids: {}` and a one-line `notes`
  explaining what's missing.
- `make golden-validate` runs schema + coverage checks and exits 0.

### Checkpoint
```bash
make golden-validate
```
**Expected:**
```
✓ schema valid
✓ 75 entries (45 factual / 15 multi_hop / 15 unanswerable)
✓ all gold_chunk_ids resolve for fixed, recursive, semantic
✓ no gold_answer > 2 sentences
```

### Risks
- Cross-strategy chunk ID mapping is manual and tedious. Mitigation:
  `05_seed_golden.py` proposes mappings by text overlap; human confirms.

---

## Phase 8 — Evaluation Metrics ⚪

**Goal:** All metrics implemented, unit-tested on hand-crafted cases.

### Deliverables
```
src/docintel/evaluation/__init__.py
src/docintel/evaluation/metrics/__init__.py
src/docintel/evaluation/metrics/base.py              # Metric ABC
src/docintel/evaluation/metrics/retrieval_precision.py
src/docintel/evaluation/metrics/answer_accuracy.py
src/docintel/evaluation/metrics/hallucination.py
src/docintel/evaluation/metrics/faithfulness.py
src/docintel/evaluation/metrics/refusal.py
src/docintel/evaluation/judge.py                     # LLM judge wrapper
src/docintel/evaluation/judge_prompts/
    correctness_v1.txt
    claim_decomposition_v1.txt
    entailment_v1.txt
    refusal_detection_v1.txt
src/docintel/evaluation/evaluator.py                 # orchestrator
tests/test_metrics.py
tests/fixtures/metrics_cases.json                    # hand-crafted positives+negatives
```

### Acceptance criteria
- Every metric in `06_EVAL_SPEC.md §6` is implemented and unit-tested.
- Test fixtures cover: perfectly correct answer, one fabricated number,
  correct refusal, incorrect refusal, gold at rank 1, gold at rank 4,
  zero-chunk retrieval, zero-claim answer.
- Judge calls are cached by `hash(prompt + model + prompt_version)`.
- `make eval-single` runs 1 config over the golden set, writes
  `results/runs/{run_id}.json` with full `EvalResult`.
- Aggregate metrics match hand-computed values on the 8 fixture questions
  within 1e-6.

### Checkpoint
```bash
make eval-single
```
**Expected:** `results/runs/{run_id}.json` with `n_questions=75`,
`aggregate_metrics` containing all 10 fields, `total_cost_usd < $0.20`.

### Risks
- Judge nondeterminism: mitigated by `temperature=0` + caching. Even so,
  re-running with cache cleared may drift ≤ 2%. Documented; report includes
  judge version.

---

## Phase 9 — Experiment Matrix & Reporting ⚪

**Goal:** One command → 9 configs → comparison report with charts.

### Deliverables
```
src/docintel/pipeline/experiment.py             # matrix runner
src/docintel/evaluation/report.py               # Markdown + CSV + PNG
src/docintel/evaluation/stats.py                # bootstrap CI
scripts/07_run_eval_matrix.py
scripts/08_generate_report.py
tests/test_stats.py
```

### Acceptance criteria
- `make eval-matrix` runs all 3×3 configs against the golden set.
- Each config produces `results/runs/{run_id}.json`.
- `results/reports/comparison.md` contains a 9-row table with all aggregate
  metrics + 95% CI half-widths + cost + latency.
- `results/figures/{timestamp}_heatmap.png` (chunking × embedding → selectable
  metric; default answer_accuracy).
- `results/figures/{timestamp}_hallucination_bar.png`.
- `results/figures/{timestamp}_cost_quality_scatter.png`.
- `results/reports/comparison.csv` for downstream analysis.
- Total cost for the matrix run logged; run aborts if it would exceed
  `budget_usd`.

### Checkpoint
```bash
make eval-matrix report
```
**Expected:** 9 run JSONs; `comparison.md` with 9 rows; 3 PNGs; CSV; total
cost printed. Runtime < 30 min warm-cache.

### Risks
- Sequential 9×75 generation calls = 675 LLM calls. At ~1.5 s each = ~17 min.
  Parallelize per config (safe; no shared write path) if under budget.
  Guard with a semaphore; default concurrency = 1.

---

## Phase 10 — API & UI ⚪

**Goal:** FastAPI endpoints + Streamlit 3-tab app.

### Deliverables
```
api/__init__.py
api/main.py
api/schemas.py
api/routes/__init__.py
api/routes/ingest.py
api/routes/query.py
api/routes/eval.py
api/deps.py                                    # config + pipeline singletons
ui/__init__.py
ui/app.py
ui/components/chat_tab.py
ui/components/eval_tab.py
ui/components/runs_tab.py
ui/charts.py                                   # plotly helpers
tests/test_api.py
```

### Acceptance criteria
- `GET /health` returns `{status, version, git_sha}`.
- `POST /ingest` accepts multipart files; returns per-file summary.
- `POST /query` returns `RAGResponse` with citations + retrieved chunks.
- `POST /eval/run` enqueues a matrix run; returns `{run_id, status}`.
- `GET /eval/runs` lists past runs (from `results/runs/`).
- `GET /eval/runs/{id}` returns full `EvalResult`.
- `GET /eval/runs/{id}/report` returns the Markdown report.
- Streamlit: 3 tabs render; chat works end-to-end; dashboard renders 3 charts;
  runs tab lists and drills into runs.

### Checkpoint
```bash
make serve
```
**Expected:** FastAPI at :8000 with `/docs` reachable; Streamlit at :8501;
all 3 tabs functional; `/health` returns 200.

### Risks
- Streamlit + FastAPI both mounting `data/` — ensure no concurrent writes.
  Eval runs are queued, not run inline.

---

## Phase 11 — Hardening ⚪

**Goal:** Tests green, Docker works from clean clone, docs complete.

### Deliverables
```
tests/test_api_integration.py
tests/test_pipeline_integration.py
tests/mocks/llm.py                              # deterministic fake LLM
docker-compose.yml                              # finalized
Dockerfile
README.md                                       # results table + diagram
CHANGELOG.md                                    # backfilled
```

### Acceptance criteria
- `pytest -q` all green.
- Coverage ≥ 80% overall; ≥ 95% on `evaluation/metrics/`.
- `docker compose up` from a clean clone reaches a working demo (both services
  healthy) in < 10 min.
- README contains: TL;DR, architecture diagram, results table, cost table,
  quick start, and the 30-second pitch.
- `CHANGELOG.md` has entries for all 11 phases.

### Checkpoint
```bash
make test
docker compose up -d && sleep 15 && curl localhost:8000/health
```
**Expected:** pytest exit 0; curl returns 200 with valid JSON.

### Risks
- Docker build time with `unstructured` + torch can exceed 15 min. Mitigate
  with multi-stage build and CPU-only torch wheels.

---

## Phase dependency graph

```
Phase 1 (scaffolding)
   │
   ▼
Phase 2 (ingestion)
   │
   ▼
Phase 3 (chunking)
   │
   ▼
Phase 4 (embeddings)
   │
   ▼
Phase 5 (index + retrieval)
   │
   ▼
Phase 6 (generation)         Phase 7 (golden dataset) ──┐
   │                                 │                  │
   └────────────────┬────────────────┘                  │
                    ▼                                   │
              Phase 8 (metrics) ◄────────────────────────┘
                    │
                    ▼
              Phase 9 (matrix + report)
                    │
                    ▼
              Phase 10 (API + UI)
                    │
                    ▼
              Phase 11 (hardening)
```

**Note:** Phase 7 can run in parallel with Phase 6 if you're blocked on
LLM API access — the golden set doesn't depend on the pipeline.