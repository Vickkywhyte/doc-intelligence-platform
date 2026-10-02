# Testing Strategy

**Related docs:** `04_ROADMAP.md`, `06_EVAL_SPEC.md`

---

## 1. Test pyramid

```
                    ▲
                   ╱ ╲      Eval smoke (real LLM, marked)
                  ╱   ╲     ~3 tests · slow · nightly/manual
                 ╱─────╲
                ╱       ╲   Integration (mocked LLM)
               ╱         ╲  ~15 tests · seconds · every commit
              ╱───────────╲
             ╱             ╲ Unit (pure functions)
            ╱               ╲ ~70 tests · milliseconds · every save
           ╱─────────────────╲
```

---

## 2. Unit tests (fast, no network, no I/O)

**Rules:**
- < 1 s per test.
- No network calls.
- No file system writes (use `tmp_path` fixture if unavoidable).
- No LLM calls.

### 2.1 Coverage by module

| Module | What to test |
|---|---|
| `utils/config.py` | Load, validation, hash stability, error on malformed |
| `utils/cache.py` | Hit/miss, key determinism, `.npy` round-trip |
| `utils/cost_tracker.py` | Accumulation, budget check, JSONL append |
| `ingestion/loader.py` | Text extraction, page count, edge cases (empty PDF) |
| `ingestion/cleaner.py` | Whitespace collapse, dehyphenation, header/footer strip |
| `chunking/*` | Offsets valid, no data loss, overlap, chunk_id format |
| `embeddings/*` | Shape, dim, batching, L2 normalization (mocked) |
| `vectorstore/*` | Upsert, query, filter, count (in-memory instance) |
| `retrieval/retriever.py` | Top-k ordering, MMR diversity |
| `generation/prompt_templates.py` | Rendering, missing var raises |
| `generation/citation_parser.py` | Extract valid, ignore malformed, edge cases |
| `evaluation/metrics/*` | Every fixture from `06_EVAL_SPEC.md §8` |
| `evaluation/judge.py` | Prompt hash stability, cache hit (mocked LLM) |
| `evaluation/stats.py` | Bootstrap CI on known distributions |

### 2.2 Critical fixtures (must exist)

- **Chunking:** sample doc with known text → assert chunk boundaries.
- **Metrics:** `tests/fixtures/metrics_cases.json` covers all §8 fixtures
  from `06_EVAL_SPEC.md`.
- **Citation parser:** 10 valid, 5 malformed, 3 near-miss citations.
- **Config hash:** same YAML from different paths → same hash; different
  YAML → different hash.

### 2.3 What NOT to unit test

- Third-party library internals (Chroma, unstructured, sentence-transformers).
- LLM output exact strings (nondeterministic).
- Streamlit rendering (visual; use manual smoke).
- Plotly figure internals (assert data, not rendering).

---

## 3. Integration tests (mocked LLM, full pipeline)

**Rules:**
- Use `tests/mocks/llm.py` — a deterministic fake LLM.
- Fast (< 5 s total for the whole integration suite).
- Exercise the full pipeline shape, not metric correctness.

### 3.1 Scenarios

| Test | What it verifies |
|---|---|
| `test_pipeline_end_to_end` | Ingest → chunk → embed (fake) → index → query → response shape valid |
| `test_evaluator_full_run` | 3-item fixture golden, 1 config, produces complete `EvalResult` |
| `test_api_health` | `GET /health` returns 200 + correct schema |
| `test_api_query` | `POST /query` with mocked pipeline returns valid `RAGResponse` |
| `test_api_eval_run` | `POST /eval/run` enqueues and returns 202 |
| `test_api_runs_list` | `GET /eval/runs` returns list from fixture runs |
| `test_cache_hit_second_run` | Embedding cache 100% on 2nd pass |
| `test_judge_cache` | Same prompt → 0 additional judge calls |
| `test_budget_abort` | Cost tracker aborts when `budget_usd` exceeded |
| `test_mmr_dedup` | MMR retrieval returns diverse chunks |
| `test_metadata_filter` | `where={"doc_type": "contract"}` narrows results |
| `test_refusal_path` | OOD question → `refused=True`, no citations |
| `test_report_generation` | Fixture run → comparison.md + CSV + 3 PNGs |
| `test_citation_validity` | Every citation chunk_id exists in retrieved set |
| `test_config_reload` | Config change is picked up without restart |

### 3.2 Mock LLM behavior

`tests/mocks/llm.py` provides:

```python
class FakeLLMClient(LLMClient):
    """Deterministic LLM for tests. Returns scripted responses."""

    def __init__(self, script: dict[str, str]):
        self.script = script          # prompt substring → response
        self.calls: list[str] = []    # record all prompts
        self.cost_usd = 0.0

    def generate(self, prompt, **kwargs) -> LLMResponse:
        self.calls.append(prompt)
        for key, resp in self.script.items():
            if key in prompt:
                return LLMResponse(text=resp, model="fake", ...)
        return LLMResponse(text="[NO SCRIPT]", model="fake", ...)
```

Tests load scripted prompts keyed by substring (e.g., `"CLAIM:"` → JSON
verdict).

---

## 4. Eval smoke tests (real LLM, marked, skipped by default)

**Rule:** Marked `@pytest.mark.llm`. Skipped unless `RUN_LLM_TESTS=1`.

**Purpose:** Catch integration breaks that mocks can't (e.g., prompt format
drift, provider API changes).

### 4.1 Tests

| Test | What it verifies | Cost cap |
|---|---|---|
| `test_smoke_full_eval_one_config` | 3 golden items × 1 config → non-empty `EvalResult` | $0.05 |
| `test_smoke_judge_determinism` | Same prompt twice → same verdict | $0.01 |
| `test_smoke_refusal_detected` | OOD question → `refused=True` | $0.01 |

### 4.2 CI policy

- **Default CI:** skip (`pytest -m "not llm"`).
- **Nightly CI:** run with `RUN_LLM_TESTS=1`, cost capped via env var.
- **Manual:** any developer can run locally with their own OpenRouter key.

---

## 5. Coverage targets

| Scope | Target |
|---|---|
| `src/docintel/` overall | ≥ 80% |
| `src/docintel/evaluation/metrics/` | ≥ 95% |
| `src/docintel/chunking/` | ≥ 90% |
| `src/docintel/utils/` | ≥ 90% |
| `src/docintel/api/` (via integration) | ≥ 70% |

Coverage measured with `pytest-cov`. Enforced in CI on push to `main`.

---

## 6. Test file layout

```
tests/
├── __init__.py
├── conftest.py                   # fixtures: tmp config, fake LLM, fixture docs
├── fixtures/
│   ├── sample_document.json      # one parsed doc
│   ├── sample_chunks.jsonl       # known chunks for one strategy
│   ├── golden_mini.jsonl         # 3 items (factual, multi_hop, unanswerable)
│   ├── metrics_cases.json        # all hand-crafted metric cases
│   └── citations.json            # 10 valid + 5 malformed + 3 near-miss
├── mocks/
│   ├── __init__.py
│   ├── llm.py                    # FakeLLMClient
│   └── embedder.py               # deterministic hash-based embedder
├── test_config.py
├── test_cache.py
├── test_cost_tracker.py
├── test_ingestion.py
├── test_chunking.py
├── test_embeddings.py
├── test_retrieval.py
├── test_prompt.py
├── test_citation_parser.py
├── test_metrics.py               # ← the critical one
├── test_stats.py
├── test_golden_schema.py
├── test_pipeline_integration.py
├── test_api.py
├── test_report.py
└── test_llm_smoke.py             # marked @pytest.mark.llm
```

---

## 7. Running tests

```bash
# Unit + integration (fast, no network)
pytest -m "not llm" -q

# With coverage
pytest -m "not llm" --cov=src/docintel --cov-report=term-missing

# Only metrics (during Phase 8)
pytest tests/test_metrics.py -v

# LLM smoke (costs money; requires OpenRouter key)
RUN_LLM_TESTS=1 pytest -m llm -v
```

### Makefile targets
```makefile
test:        ## Unit + integration, no LLM
	pytest -m "not llm" -q

test-cov:    ## With coverage report
	pytest -m "not llm" --cov=src/docintel --cov-report=term-missing

test-metrics: ## Only metric unit tests
	pytest tests/test_metrics.py -v

test-llm:    ## Smoke tests with real LLM (costs money)
	RUN_LLM_TESTS=1 pytest -m llm -v
```

---

## 8. CI pipeline (documented; can be enabled with GitHub Actions)

```yaml
# .github/workflows/ci.yml (documented, not built in v1)
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: pytest -m "not llm" --cov=src/docintel --cov-fail-under=80
```

**Nightly** adds an `llm-smoke` job gated on a repository secret for the
OpenRouter key, with `--budget-usd 0.25`.

---

## 9. Testing philosophy

1. **Test the metric definitions, not the LLM's output.** The metric is code;
   the LLM is a black box.
2. **Prefer fixtures over mocks.** Real chunks, real citations, real config
   → catch integration bugs early.
3. **Mock at the boundary, not in the middle.** `FakeLLMClient` implements
   `LLMClient` ABC — same interface as production.
4. **No flaky tests.** Nondeterminism only in `@pytest.mark.llm` tests.
5. **Fast tests are run tests.** If the unit suite takes > 10 s, it's too slow.
6. **Coverage is a floor, not a goal.** 95% on metrics because they're
   correctness-critical; 70% on API because integration tests cover it.