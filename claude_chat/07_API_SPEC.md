# API Specification

**Base URL:** `http://localhost:8000`
**OpenAPI:** `http://localhost:8000/docs`
**Related docs:** `05_DATA_MODEL.md`, `08_UI_SPEC.md`

---

## 1. Conventions

- **Content type:** `application/json` (except `/ingest` which is multipart).
- **Timestamps:** ISO 8601 with timezone (UTC).
- **Errors:** Consistent envelope (§7).
- **Auth (v1):** None. Local-only default. Documented in `11_SECURITY_PRIVACY.md`.
- **Rate limiting:** None (local).
- **Versioning:** Path-prefixed `/v1/` reserved for future; v1 endpoints
  live at root.

---

## 2. Endpoints summary

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + version |
| `POST` | `/ingest` | Upload + ingest files |
| `POST` | `/query` | Ask a question |
| `POST` | `/eval/run` | Enqueue a matrix run |
| `GET` | `/eval/runs` | List runs (summaries) |
| `GET` | `/eval/runs/{run_id}` | Full EvalResult |
| `GET` | `/eval/runs/{run_id}/report` | Markdown report |
| `GET` | `/eval/runs/{run_id}/figures/{name}` | PNG figure |

---

## 3. `GET /health`

**Purpose:** Liveness + version probe.

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "git_sha": "f3a9b21",
  "python_version": "3.11.9",
  "uptime_s": 421
}
```

**Response 503** (if a critical dependency is down):
```json
{"status": "degraded", "reason": "vector_store_unreachable"}
```

---

## 4. `POST /ingest`

**Purpose:** Upload one or more files and ingest them into the corpus.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `files` | file[] | yes | One or more PDF/DOCX/TXT/MD |
| `doc_type` | string | no | Default `other`; must be a valid `DocType` |

**Response 200:**
```json
{
  "processed": [
    {
      "doc_id": "a1b2c3d4e5f6a7b8",
      "filename": "msa_2024.pdf",
      "doc_type": "contract",
      "pages": 12,
      "chars": 45231,
      "chunks": {"fixed": 88, "recursive": 71, "semantic": 54}
    }
  ],
  "failed": [
    {"filename": "scanned_scan.pdf", "error": "OCRRequired: no extractable text"}
  ],
  "summary": {"n_processed": 1, "n_failed": 1, "total_chars": 45231}
}
```

**Notes:**
- Chunks computed for all three strategies in one call.
- Embeddings NOT computed here (separate concern; use `make embed` or a
  future `/index` endpoint). Documented behavior.

**Errors:**
- `400 INVALID_DOC_TYPE` — bad `doc_type` value.
- `400 NO_FILES` — empty `files` field.
- `413 FILE_TOO_LARGE` — any file > `config.ingestion.max_file_mb`.
- `422 INGEST_FAILED` — all files failed (per-file errors in `failed`).

---

## 5. `POST /query`

**Purpose:** Ask a question against the indexed corpus.

**Request:**
```json
{
  "question": "What is the notice period for termination without cause?",
  "config_override": {
    "retrieval": {"k": 8, "mmr": true}
  },
  "return_chunks": true
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | 1–2000 chars |
| `config_override` | object | no | Partial override; validated against schema |
| `return_chunks` | bool | no | Default `true` |

**Response 200:**
```json
{
  "query": "What is the notice period for termination without cause?",
  "answer": "30 days written notice [a1b2c3d4e5f6a7b8:7:a1b2c3d4e5f6a7b8:recursive:00012].",
  "refused": false,
  "citations": [
    {"doc_id": "a1b2c3d4e5f6a7b8", "page": 7,
     "chunk_id": "a1b2c3d4e5f6a7b8:recursive:00012"}
  ],
  "retrieved": [
    {"chunk_id": "a1b2c3d4e5f6a7b8:recursive:00012",
     "score": 0.83, "rank": 1, "retriever": "topk",
     "text": "Either party may terminate this Agreement without cause upon thirty (30) days' prior written notice...",
     "page": 7}
  ],
  "model": "openai/gpt-4o-mini",
  "latency_ms": 1240,
  "cost_usd": 0.0004,
  "warnings": []
}
```

**Response 200 (refusal):**
```json
{
  "query": "What is our policy on cryptocurrency reimbursement?",
  "answer": "I don't know based on the provided documents.",
  "refused": true,
  "citations": [],
  "retrieved": [
    {"chunk_id": "...", "score": 0.21, "rank": 1, "retriever": "topk",
     "text": "...", "page": 3}
  ],
  "model": "openai/gpt-4o-mini",
  "latency_ms": 890,
  "cost_usd": 0.0003,
  "warnings": ["low_retrieval_score"]
}
```

**Errors:**
- `400 EMPTY_QUESTION` — blank question.
- `400 QUESTION_TOO_LONG` — > 2000 chars.
- `503 INDEX_MISSING` — no index exists for the current config.
- `504 LLM_TIMEOUT` — provider timeout after retries.

---

## 6. `POST /eval/run`

**Purpose:** Enqueue a full matrix run. Runs asynchronously.

**Request:**
```json
{
  "config_path": "configs/experiment_matrix.yaml",
  "notes": "baseline v1",
  "subset": null
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `config_path` | string | no | Default canonical config |
| `notes` | string | no | Free-text annotation on the run |
| `subset` | string[] | no | Golden IDs to run (for debugging) |

**Response 202:**
```json
{
  "run_id": "20250115T103000_a1b2c3d4",
  "status": "queued",
  "n_configs": 9,
  "n_questions": 75,
  "estimated_cost_usd": 0.45,
  "estimated_duration_s": 1080
}
```

**Errors:**
- `404 CONFIG_NOT_FOUND` — `config_path` missing.
- `400 BUDGET_EXCEEDED` — estimated cost > `budget_usd`.
- `409 RUN_IN_PROGRESS` — only one matrix run at a time.

---

## 7. `GET /eval/runs`

**Purpose:** List past runs, newest first.

**Query params:**
| Param | Type | Default | Notes |
|---|---|---|---|
| `limit` | int | 20 | Max 100 |
| `status` | string | `completed` | `completed` / `failed` / `running` / `all` |

**Response 200:**
```json
{
  "runs": [
    {
      "run_id": "20250115T103000_a1b2c3d4",
      "config_hash": "a1b2c3d4e5f6a7b8...",
      "chunking": "recursive",
      "embedding": "bge_small_en_v15",
      "started_at": "2025-01-15T10:30:00Z",
      "ended_at":   "2025-01-15T10:47:12Z",
      "n_questions": 75,
      "best_metric": {"name": "answer_accuracy", "value": 0.87},
      "total_cost_usd": 0.031,
      "status": "completed"
    }
  ],
  "total": 9
}
```

**Note:** `chunking` / `embedding` shown here are **the config's first entry**
when a run contains multiple configs. For multi-config runs, expand per-config
rows via `/eval/runs/{run_id}`.

---

## 8. `GET /eval/runs/{run_id}`

**Purpose:** Full `EvalResult` for a run.

**Response 200:** The `EvalResult` model (§`05_DATA_MODEL.md` §2.5).

**Errors:**
- `404 RUN_NOT_FOUND`.

---

## 9. `GET /eval/runs/{run_id}/report`

**Purpose:** Rendered Markdown comparison report.

**Query params:**
| Param | Type | Default | Notes |
|---|---|---|---|
| `format` | string | `md` | `md` / `html` |

**Response 200:** `text/markdown` or `text/html`.

**Errors:**
- `404 RUN_NOT_FOUND`.
- `404 REPORT_NOT_READY` — run still in progress.

---

## 10. `GET /eval/runs/{run_id}/figures/{name}`

**Purpose:** Serve a generated PNG.

**Path params:**
- `name` ∈ {`heatmap`, `hallucination_bar`, `cost_quality_scatter`}

**Response 200:** `image/png`

**Errors:**
- `404 FIGURE_NOT_FOUND`.

---

## 11. Error envelope

All 4xx/5xx responses share this shape:

```json
{
  "error": {
    "code": "EMPTY_QUESTION",
    "message": "Question must be at least 1 character.",
    "detail": {"field": "question", "received_length": 0},
    "request_id": "req_01HZX..."
  }
}
```

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `INVALID_DOC_TYPE` | Bad enum value |
| 400 | `NO_FILES` | Empty multipart |
| 400 | `EMPTY_QUESTION` | Blank question |
| 400 | `QUESTION_TOO_LONG` | > 2000 chars |
| 400 | `BUDGET_EXCEEDED` | Estimated cost exceeds cap |
| 404 | `CONFIG_NOT_FOUND` | Config path missing |
| 404 | `RUN_NOT_FOUND` | Unknown run_id |
| 404 | `REPORT_NOT_READY` | Run in progress |
| 409 | `RUN_IN_PROGRESS` | Only one matrix run at a time |
| 413 | `FILE_TOO_LARGE` | Exceeds `max_file_mb` |
| 422 | `INGEST_FAILED` | All files failed |
| 500 | `INTERNAL_ERROR` | Uncaught; logged with `request_id` |
| 503 | `INDEX_MISSING` | No index for current config |
| 503 | `DEGRADED` | Dependency down |
| 504 | `LLM_TIMEOUT` | Provider timeout after retries |

---

## 12. Examples

### 12.1 Full flow with `curl`

```bash
# Health
curl -s localhost:8000/health | jq

# Ingest
curl -s -X POST localhost:8000/ingest \
  -F "files=@data/raw/msa_2024.pdf" \
  -F "doc_type=contract" | jq

# Query
curl -s -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the notice period for termination?"}' | jq

# Run matrix
curl -s -X POST localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"notes":"baseline v1"}' | jq

# Poll
curl -s localhost:8000/eval/runs | jq '.runs[0]'

# Report
curl -s localhost:8000/eval/runs/20250115T103000_a1b2c3d4/report
```

### 12.2 Python client

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as c:
    r = c.post("/query", json={"question": "What is the notice period?"})
    r.raise_for_status()
    print(r.json()["answer"])
```

---

## 13. Non-goals for v1 API

- Streaming responses (SSE / WebSocket)
- File deletion endpoint
- User accounts, sessions, RBAC
- Webhooks on run completion
- GraphQL

These are documented as v2 considerations.