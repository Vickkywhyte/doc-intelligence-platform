# Observability

**Related docs:** `02_ARCHITECTURE.md`, `05_DATA_MODEL.md`, `06_EVAL_SPEC.md`

---

## 1. Principles

1. **Structured, not printed.** All logs are JSON via `structlog`.
2. **Cost is a first-class signal.** Every LLM/embedding call emits a cost event.
3. **Reproducibility over richness.** Every run is fully reconstructible
   from its artifact.
4. **No PII in logs.** Documents are referenced by `doc_id`, not filename
   when sensitive.
5. **Queryable on disk.** Logs and run artifacts are JSON/JSONL — greppable
   with `jq`, loadable in a notebook.

---

## 2. Logging

### 2.1 Library

`structlog` with JSON renderer. Configured once in `utils/logging.py`.

```python
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)
```

### 2.2 Standard fields

Every log line includes:

| Field | Source | Notes |
|---|---|---|
| `ts` | auto | ISO 8601 UTC |
| `level` | auto | `info` / `warning` / `error` |
| `event` | caller | snake_case, e.g. `ingest_document_complete` |
| `run_id` | contextvar | set during eval runs |
| `doc_id` | caller | when applicable |
| `chunk_id` | caller | when applicable |
| `latency_ms` | caller | for timed operations |
| `cost_usd` | caller | for LLM/embedding calls |

`run_id` is bound once at run start via `structlog.contextvars.bind_contextvars(run_id=...)` and merged into every subsequent log line in that run.

### 2.3 Naming convention

`{domain}_{object}_{action}` — e.g.:
- `ingest_document_start` / `ingest_document_complete` / `ingest_document_failed`
- `chunk_strategy_start` / `chunk_strategy_complete`
- `embed_batch_start` / `embed_batch_complete` / `embed_cache_hit`
- `retrieval_query_start` / `retrieval_query_complete`
- `llm_call_start` / `llm_call_complete` / `llm_call_failed`
- `judge_call_start` / `judge_call_complete` / `judge_cache_hit`
- `eval_run_start` / `eval_run_complete` / `eval_run_aborted`
- `cost_budget_warning` / `cost_budget_exceeded`

### 2.4 Destinations

| Sink | Purpose | Path |
|---|---|---|
| stdout | dev console | — |
| rotating file | post-hoc analysis | `logs/app.jsonl` (10 MB × 5 files) |

Rotation via `logging.handlers.RotatingFileHandler`.

### 2.5 Example log lines

```json
{"ts":"2025-01-15T10:30:00.123Z","level":"info","event":"ingest_document_complete","doc_id":"a1b2c3d4e5f6a7b8","filename":"msa_2024.pdf","pages":12,"chars":45231,"latency_ms":847}
{"ts":"2025-01-15T10:30:01.456Z","level":"info","event":"chunk_strategy_complete","doc_id":"a1b2c3d4e5f6a7b8","strategy":"recursive","n_chunks":71,"avg_tokens":118,"latency_ms":12}
{"ts":"2025-01-15T10:30:05.789Z","level":"info","event":"embed_cache_hit","model":"bge_small_en_v15","n_texts":71,"latency_ms":3}
{"ts":"2025-01-15T10:30:20.012Z","level":"info","event":"llm_call_complete","model":"openai/gpt-4o-mini","prompt_tokens":842,"completion_tokens":67,"cost_usd":0.00021,"latency_ms":1340,"run_id":"20250115T103000_a1b2c3d4"}
{"ts":"2025-01-15T10:30:22.345Z","level":"warning","event":"retrieval_low_score","query_id":"q042","top_score":0.21,"threshold":0.35}
{"ts":"2025-01-15T10:30:22.678Z","level":"error","event":"ingest_document_failed","filename":"scanned.pdf","error":"OCRRequired: no extractable text","traceback":"..."}
```

---

## 3. Cost tracking

### 3.1 Ledger

Append-only JSONL at `logs/cost.jsonl`. One line per billable call:

```json
{"ts":"2025-01-15T10:30:20.012Z","run_id":"20250115T103000_a1b2c3d4","kind":"llm","model":"openai/gpt-4o-mini","prompt_tokens":842,"completion_tokens":67,"cost_usd":0.00021}
{"ts":"2025-01-15T10:30:20.345Z","run_id":"20250115T103000_a1b2c3d4","kind":"judge","model":"openai/gpt-4o-mini","prompt_tokens":512,"completion_tokens":23,"cost_usd":0.00009}
{"ts":"2025-01-15T10:30:21.012Z","run_id":"20250115T103000_a1b2c3d4","kind":"embedding","model":"openai/text-embedding-3-small","tokens":1804,"cost_usd":0.00004}
```

### 3.2 Price table

`configs/prices.yaml` — versioned, dated, easy to update. Model names use
OpenRouter's `provider/model` format:

```yaml
# Prices in USD per 1M tokens. Update when providers change rates.
# Last updated: 2025-01-10
# NOTE: Prices reflect OpenRouter list prices (which mirror provider pricing).
llm:
  openai/gpt-4o-mini:
    prompt: 0.15
    completion: 0.60
  openai/gpt-4o:
    prompt: 2.50
    completion: 10.00
  anthropic/claude-3.5-haiku:
    prompt: 0.80
    completion: 4.00
  anthropic/claude-3.5-sonnet:
    prompt: 3.00
    completion: 15.00
embeddings:
  openai/text-embedding-3-small:
    tokens: 0.02
  openai/text-embedding-3-large:
    tokens: 0.13
local:
  bge_small_en_v15:
    tokens: 0.0
  minilm_l6_v2:
    tokens: 0.0
```

### 3.3 Budget guard

- Config field `evaluation.budget_usd` (default $5.00).
- `CostTracker` checks after every billable call:
  - ≥ 80% → `cost_budget_warning` log
  - ≥ 100% → `cost_budget_exceeded` log + `BudgetExceededError` raised
- Long-running runs (matrix) abort cleanly, persist partial `EvalResult` to
  `results/runs/{id}.partial.json`.

### 3.4 Session totals

`CostTracker` also maintains a process-local counter for UI display
("Session cost: $0.0142").

---

## 4. Run artifacts

Every eval run produces:

| Artifact | Path | Content |
|---|---|---|
| Full result | `results/runs/{run_id}.json` | Complete `EvalResult` |
| Partial (if aborted) | `results/runs/{run_id}.partial.json` | Partial results up to abort |
| Report | `results/reports/{run_id}.md` | Human-readable comparison |
| CSV | `results/reports/{run_id}.csv` | Flat per-config metrics |
| Heatmap | `results/figures/{run_id}_heatmap.png` | Plotly PNG |
| Hallucination bar | `results/figures/{run_id}_hallucination_bar.png` | Plotly PNG |
| Cost-quality scatter | `results/figures/{run_id}_cost_quality_scatter.png` | Plotly PNG |

### 4.1 Reproducibility fields (mandatory in every `EvalResult`)

- `run_id` — `{timestamp}_{config_hash[:8]}`
- `config_hash` — SHA-256 of canonicalized YAML
- `config` — full resolved config
- `git_sha` — current commit (or `null` if unavailable)
- `python_version` — `sys.version_info`
- `package_versions` — key deps from `importlib.metadata`
- `judge_model`, `judge_prompt_version`
- `seed`
- `started_at`, `ended_at`
- `total_cost_usd`

### 4.2 Nondeterminism disclosure

If a provider does not support `seed=` on generation, log a warning and add
to `EvalResult.notes`:
```
"Nondeterminism: provider does not honor seed; expect ±1-2% drift."
```

---

## 5. Metrics emitted per run

Beyond the eval metrics in `06_EVAL_SPEC.md §6`, the following operational
metrics are logged and stored in `EvalResult.config` (as run metadata):

| Metric | Source |
|---|---|
| `n_llm_calls` | Count of billable generation calls |
| `n_judge_calls` | Count of billable judge calls |
| `n_embedding_calls` | Count of billable embedding calls (cache misses) |
| `cache_hit_rate_embeddings` | Fraction of embeddings from cache |
| `cache_hit_rate_judge` | Fraction of judge calls from cache |
| `wall_time_s` | `ended_at - started_at` |
| `time_per_question_ms` | `wall_time_s * 1000 / n_questions` |
| `errors` | List of `{event, doc_id?, message}` |

---

## 6. Debugging workflows

### 6.1 "Why is this config slow?"

```bash
jq 'select(.event=="llm_call_complete") | .latency_ms' logs/app.jsonl \
  | sort -n | tail -20
```

### 6.2 "Which questions failed retrieval?"

```bash
jq '.per_question[] | select(.metrics.p_at_5 == 0)
    | {id: .item_id, q: .question}' results/runs/20250115T103000_a1b2c3d4.json
```

### 6.3 "How much did today cost?"

```bash
jq -s 'map(.cost_usd) | add' logs/cost.jsonl
```

### 6.4 "What's in this run's config?"

```bash
jq '.config' results/runs/20250115T103000_a1b2c3d4.json
```

### 6.5 "Show me all hallucinated claims"

```bash
jq '.per_question[] | {id: .item_id, claims: .hallucinated_claims}
    | select(.claims | length > 0)' results/runs/*.json
```

---

## 7. What we deliberately do NOT build (v1)

- **No OpenTelemetry.** Structured JSON logs + JSONL ledger suffice.
- **No Prometheus/Grafana.** Local-first project; no metrics server.
- **No Sentry.** Errors are logged with tracebacks; no remote reporting.
- **No log shipping.** Logs stay on disk.
- **No tracing spans.** `run_id` + `doc_id` + `chunk_id` correlate events;
  full distributed tracing is overkill for a monolith.

These become candidates if the project is deployed multi-user.

---

## 8. Retention

| Artifact | Retention (default) |
|---|---|
| `logs/app.jsonl` | Rolling: 10 MB × 5 files |
| `logs/cost.jsonl` | Unlimited (tiny; ~100 bytes/line) |
| `results/runs/*.json` | Unlimited (gitignored; manual cleanup) |
| `results/figures/*.png` | Unlimited |
| `data/cache/embeddings/` | Manual (`make clean-cache`) |
| `data/cache/judge/` | Manual |

`make clean` targets documented in `Makefile`.

---

## 9. The "reproducibility receipt"

Every report (`results/reports/{run_id}.md`) begins with:

```
## Run metadata

- Run ID:            20250115T103000_a1b2c3d4
- Config hash:       a1b2c3d4e5f6a7b8...
- Git SHA:           f3a9b21
- Python:            3.11.9
- Judge:             openai/gpt-4o-mini (prompt v1, temp=0)
- Generator:         openai/gpt-4o-mini (temp=0)
- Seed:              0
- Nondeterminism:    none declared
- Started:           2025-01-15T10:30:00Z
- Ended:             2025-01-15T10:47:12Z
- Total cost:        $0.31
- Cache hit rate:    embeddings 0.00 · judge 0.00
```

This block is generated by `report.py` and is the first thing a reviewer sees.
It answers "can I trust this number?" in six lines.