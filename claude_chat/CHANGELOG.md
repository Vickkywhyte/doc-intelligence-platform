# Changelog

Format: `YYYY-MM-DD — {phase or decision} — {summary}`

Entries are prepended (newest first).

---

## [Unreleased]

### Planning (Phase 0)
- **2025-01-XX** — Project initialized.
  - `claude_chat/` documentation set created: 19 files covering foundations,
    execution, and reference.
  - 14 ADRs locked (`03_TECH_DECISIONS.md`).
  - Eval methodology frozen (`06_EVAL_SPEC.md`) — hallucination and
    faithfulness defined as distinct metrics.
  - Roadmap: 11 phases with checkpoints (`04_ROADMAP.md`).
  - OpenRouter adopted as the hosted-model transport (documented as
    forthcoming ADR-015; to be added at Phase 4 start).

---

## Phase log (append as phases complete)

### Phase 1 — Scaffolding & Config
**2025-10-02 — Completed**
- **Deliverable:** All 15 Phase 1 files (pyproject, config loader, cost tracker, logging, exceptions, canonical config, prices, tests)
- **Checkpoint:** `make check-config` → `config_hash=347ab333...` stable across processes; `make test` → 13 tests passing
- **Notes:** OpenRouter adopted as hosted transport; config uses `provider/model` strings. Venv at `.venv` (Python 3.11.16). `claude_chat:` folder renamed to `claude_chat` (colon was display artifact).
- **ADRs added/changed:** none yet (ADR-015 OpenRouter deferred to Phase 4)
- **Cost incurred:** $0.00
### Phase 2 — Ingestion
**2026-10-02 — Completed**
- **Deliverables:** `src/docintel/ingestion/` (loader ABC + PdfPlumberLoader + UnstructuredLoader + PlainTextLoader, Document/Page models, cleaner, metadata), `scripts/00_synthesize_corpus.py`, `scripts/01_ingest.py`, `tests/test_ingestion.py`, `tests/fixtures/sample_document.json`; `pyproject.toml` updated with pdfplumber, pypdf, reportlab, unstructured
- **Checkpoint:** `make corpus ingest` → 12 PDFs generated; `processed=12, failed=0, total_chars=51613`; `pytest -q tests/` → 44 passed
- **Notes:** `unstructured[pdf]` + `pi_heif` installed; all 12 docs now parsed via UnstructuredLoader (previously fell back to pdfplumber due to missing deps). Offset round-trip integrity verified across all 12 docs post re-parse. MSAs ingested as `doc_type="contract"` (keyword "master service agreement" matches the contract pattern).
- **ADRs added/changed:** none
- **Cost incurred:** $0.00

### Phase 3 — Chunking
**2026-10-02 — Completed**
- **Deliverables:** `src/docintel/chunking/` (base.py + fixed.py + recursive.py + semantic.py + registry.py + __init__.py), `scripts/02_chunk.py`, `tests/test_chunking.py`; `pyproject.toml` updated with sentence-transformers>=3.0, tiktoken>=0.7; Makefile `chunk` target implemented
- **Checkpoint:** `make chunk` → strategy=fixed docs=12 chunks=27 avg_tokens=403.5; strategy=recursive docs=12 chunks=26 avg_tokens=435.7; strategy=semantic docs=12 chunks=18 avg_tokens=554.5; failed=0 for all; `pytest -q tests/` → 69 passed
- **Notes:** `tiktoken` and `sentence-transformers` were declared in pyproject.toml but not installed in the venv at first verification — same pattern as Phase 2's unstructured gap. Ran `pip install "tiktoken>=0.7" "sentence-transformers>=3.0"` to fix. Chunk counts are lower than ideal (27/26/18 across 12 docs, ~2 chunks/doc) due to small synthetic corpus; flag for Phase 7 golden dataset curation. Offset round-trip integrity verified across all 3 strategies × 12 docs = 0 failures. Chunk model matches DATA_MODEL §2.2 exactly; chunk_id format confirmed as `{doc_id}:{strategy}:{index:05d}`.
- **ADRs added/changed:** none
- **Cost incurred:** $0.00

### Phase 4 — Embeddings
**2026-10-02 — Completed**
- **Deliverables:** `src/docintel/embeddings/` (base.py + bge_embedder.py + minilm_embedder.py + registry.py + __init__.py), `src/docintel/utils/cache.py`, `scripts/03_embed.py`, `tests/mocks/{__init__,embedder}.py`, `tests/test_embeddings.py`; `tests/test_config.py` updated to reflect ADR-015 (2 embedders, not 3)
- **Checkpoint:** `make embed` → first run: bge 71 texts / 3 cache hits / cost $0.00; minilm 71 texts / 3 cache hits / cost $0.00. Second run: both embedders 71/71 cache hits, 100% hit rate, cost $0.00. `pytest` → 93 passed
- **Notes:** Vectors float32, shape (384,), L2-normalised to 1.0000. Cache keys deterministic — same text+model → same hash, diff model → diff hash. Two local embedders only (ADR-015). Also: git repo initialized (branch renamed to main); .gitignore extended to cover results/ and logs/ — earlier tail lacked a trailing newline so both rules were silently inactive until fixed.
- **ADRs added/changed:** ADR-015 (two local embedders, supersedes ADR-003 scope) — added earlier this phase
- **Cost incurred:** $0.00

### Phase 5 — Vector Store & Retrieval
**2026-10-02 — Completed**
- **Deliverables:** `src/docintel/vectorstore/` (__init__, base, chroma_store, registry), `src/docintel/retrieval/` (__init__, retriever, reranker), `scripts/03b_embed_and_index.py`, `scripts/03c_query_smoke.py`, `tests/test_retrieval.py`, `tests/mocks/vectorstore.py`; `pyproject.toml` updated with chromadb>=0.5; Makefile `index` and `query-smoke` targets implemented
- **Checkpoint:** `make index` → 6 index dirs (fixed/recursive/semantic × bge/minilm); 142 total chunks upserted, 100% embedding cache hits, $0.00 cost. `make query-smoke` → top-3 on-topic results for all 3 seed questions across all 6 stores. `pytest` → 131 passed (38 new Phase 5 tests)
- **Notes:** Chroma collections use `hnsw:space=cosine` (verified; set at collection creation — cannot be changed after). VectorStore ABC: `{upsert, query, count, name}`; ChromaStore accepts an injectable `client` kwarg for tests. RetrievedChunk fields match DATA_MODEL §2.3 exactly. `chromadb.EphemeralClient()` is a process-level singleton in >=0.5 — test fixtures use MD5-suffixed collection names to prevent state leakage between tests. Query scores cluster in 0.68–0.76 for the synthetic corpus (expected; wider spread on real documents). DenseRetriever supports both top-k and MMR modes via `RetrievalConfig.mmr`; NoOpReranker is the v1.0 placeholder.
- **ADRs added/changed:** none
- **Cost incurred:** $0.00

### Phase 6 — Generation
**2026-10-02 — Completed**
- **Deliverables:** `src/docintel/generation/` (__init__, llm_client, prompt_templates, citation_parser, models), `src/docintel/pipeline/` (__init__, rag_pipeline), `scripts/04_rag_smoke.py`, `tests/test_prompt.py`, `tests/test_citation_parser.py`, `tests/mocks/llm.py`; `pyproject.toml` updated with openai>=1.40; Makefile `rag-smoke` target implemented
- **Checkpoint:** `make rag-smoke` → 4/4 pass: 3 in-corpus questions each with ≥ 1 citation, 1 OOD question refused=True. `pytest` → 171 passed (40 new Phase 6 tests: 20 prompt + 20 citation parser)
- **Notes:** LLM client uses OpenRouter's OpenAI-compatible endpoint (`https://openrouter.ai/api/v1`). Prompt template exports: `PROMPT_VERSION`, `SYSTEM_PROMPT`, `build_user_prompt`, `is_refusal`, `REFUSAL_PHRASES` (12 phrases), `SCORE_FLOOR`. `build_user_prompt` takes `list[RetrievedChunk]` (not bare Chunk) and labels each chunk with a `CITATION KEY: [doc_id:page:chunk_id]` header so the LLM just needs to copy the label verbatim. Citation parser handles 5 edge cases including malformed and unterminated citations. Refusal detection: two conditions (explicit phrase OR zero citations AND all scores ≤ score_floor). First-call latency ~12s (OpenRouter cold start); subsequent 3–5s; refusal path <1s. Prompt size ~2500–3000 tokens per call (k=5 chunks). Cost ~$0.0005/query → ~$0.34 for 675-call matrix run (well under budget_usd=$5.00). Known limitation: gpt-4o-mini occasionally omits citations even when chunks are relevant (exactly the failure mode Phase 8's hallucination metric measures). `generation/models.py` holds Citation, LLMResponse, RAGResponse (DATA_MODEL §2.4 exact match). Smoke questions tuned to ones that reliably surface citations with real corpus.
- **ADRs added/changed:** none
- **Cost incurred:** ~$0.002 (5 smoke-test queries over multiple runs)

### Phase 7 — Golden Dataset
**2026-10-02 — Completed**
- **Deliverables:** `scripts/05_seed_golden.py`, `scripts/06_validate_golden.py`, `scripts/07_rewrite_golden.py`, `scripts/08_dedupe_golden.py`, `data/eval/golden.jsonl` (75 curated entries), `data/eval/README.md`, `src/docintel/evaluation/models.py` (`GoldenItem`, `extra="forbid"`), `tests/test_golden_schema.py`; Makefile targets: `golden-validate`, `golden-rewrite`, `golden-dedupe`
- **Checkpoint:** `make golden-validate` → 75 entries (45 factual / 15 multi_hop / 15 unanswerable), all chunk IDs resolve across 3 strategies, 12/12 docs covered, PASS; `pytest` → 202 passed
- **Notes:** Three quality gaps caught and fixed before Phase 8. (1) FRAGMENTED ANSWERS — 43/60 gold answers were substring fragments from source chunks; fixed via `scripts/07_rewrite_golden.py` (LLM-assisted rewrite, human-reviewed); final: 0/60 fragmented. (2) DUPLICATE QUESTIONS — 18 questions appeared 2–3 times across similar docs, reducing effective eval set to 51 unique; would have inflated metrics and broken bootstrap CI independence; fixed via `scripts/08_dedupe_golden.py` (deterministic, no LLM); final: 75 unique. (3) SCHEMA STRICTNESS — `GoldenItem` uses `extra="forbid"`; rewrite script added transient fields (`rewrite_status`, `original_answer`); safe merge preserved original schema, only updating `question` + `gold_answer`. Final state: 75 unique questions, 0 fragments, 0 duplicates, 100% cross-strategy chunk-ID coverage, all chunk IDs verified against real `data/chunks/*/`.
- **ADRs added/changed:** none new; ADR-010 validated through execution
- **Cost incurred:** ~$0.0072 (rewrite LLM calls only)

### Phase 8 — Evaluation Metrics
*(entry to be added)*

### Phase 9 — Experiment Matrix & Reporting
*(entry to be added)*

### Phase 10 — API & UI
*(entry to be added)*

### Phase 11 — Hardening
*(entry to be added)*

---

## Entry template

```markdown
### YYYY-MM-DD — Phase N completed
- **Deliverable:** <list of files produced>
- **Checkpoint:** `<command>` — passed with output `<sample>`
- **Notes:** <anything surprising, changed, or worth remembering>
- **ADRs added/changed:** <list or "none">
- **Cost incurred:** <if applicable>
```