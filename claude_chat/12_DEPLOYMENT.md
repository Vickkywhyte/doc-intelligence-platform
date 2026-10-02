# Deployment

**Related docs:** `01_PRD.md`, `11_SECURITY_PRIVACY.md`, `02_ARCHITECTURE.md`

---

## 1. Local (primary target)

### 1.1 Prerequisites

- Python 3.11
- `poppler` and `tesseract` (for `unstructured` hi_res mode)
- Optional: an OpenRouter API key (for hosted models)
- Optional: 2 GB free disk (models + indexes + corpus)

### 1.2 Quick start

```bash
git clone <repo> && cd doc-intelligence-platform
cp .env.example .env              # add OPENROUTER_API_KEY if using hosted
make install                      # pip install -e ".[dev]"
make corpus                       # generate synthetic sample PDFs
make ingest chunk embed index     # build the corpus index
make eval-matrix                  # run the full comparison (optional but recommended)
make serve                        # FastAPI :8000 + Streamlit :8501
```

Open http://localhost:8501 and click **Eval Dashboard**.

### 1.3 First-run expectations

| Step | Time (CPU laptop) |
|---|---|
| `make install` | 2–4 min (downloads deps) |
| Local model download (Phase 4) | 1–2 min (~250 MB) |
| `make corpus` | < 10 s |
| `make ingest` | < 30 s |
| `make chunk` | < 30 s |
| `make embed` (cold) | 3–5 min |
| `make embed` (warm cache) | < 5 s |
| `make index` | < 30 s |
| `make eval-matrix` | 15–30 min (hosted) |
| `make serve` | < 5 s |

---

## 2. Docker Compose (secondary target)

### 2.1 Compose file

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  ui:
    build: .
    command: >
      streamlit run ui/app.py
      --server.port 8501
      --server.address 0.0.0.0
    ports: ["8501:8501"]
    depends_on:
      api:
        condition: service_healthy
    env_file: .env
    environment:
      API_BASE_URL: http://api:8000
    volumes:
      - ./data:/app/data
      - ./results:/app/results
```

### 2.2 Dockerfile (multi-stage, CPU-only)

```dockerfile
# Stage 1: build
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e ".[dev]"

# Stage 2: runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# CPU-only torch (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.3 Commands

```bash
docker compose up -d              # start both services
docker compose logs -f api        # tail API logs
docker compose exec api bash      # shell into API container
docker compose down               # stop everything
docker compose down -v            # stop + remove volumes (data loss)
```

### 2.4 Known caveats

- **Build time:** 8–15 min on first build (torch + unstructured).
- **Volume permissions:** if `data/` is owned by root on host, container may
  write files owned by root. Fix with `chown` or use a named volume.
- **Model cache:** local models download to `~/.cache/huggingface` inside
  the container. To persist across rebuilds, add:
  ```yaml
  volumes:
    - hf_cache:/root/.cache/huggingface
  ```
  with a top-level `volumes: { hf_cache: }` declaration.

---

## 3. Cloud path (documented, not built in v1)

If adapting for multi-user deployment:

### 3.1 Architecture changes

| Concern | v1 (local) | v2 (cloud) |
|---|---|---|
| Auth | None | API keys / OAuth |
| Vector store | Local Chroma | Qdrant Cloud / pgvector |
| File storage | Local `data/` | S3 / GCS |
| Run artifacts | Local `results/` | S3 + DynamoDB / Postgres |
| Secrets | `.env` | AWS Secrets Manager / Doppler |
| Logs | Local JSONL | CloudWatch / Datadog |
| Deploy | Docker Compose | ECS / Fly.io / Render |
| Scaling | Single process | Multiple API replicas behind LB |

### 3.2 Migration path

The `VectorStore` ABC means swapping Chroma for Qdrant is:
1. `pip install qdrant-client`
2. Add `src/docintel/vectorstore/qdrant_store.py` implementing `VectorStore`
3. Register in `vectorstore/registry.py`
4. Change `configs/experiment_matrix.yaml` vector store name

No changes to `retrieval/`, `pipeline/`, or `api/`.

### 3.3 Secrets

- **AWS:** Secrets Manager + IAM role for the ECS task. No keys in env.
- **Fly.io:** `fly secrets set OPENROUTER_API_KEY=...`
- **Render:** Environment variables in the dashboard.

### 3.4 Multi-tenant considerations

Not addressed in v1. If needed:
- Per-tenant Chroma collection or Qdrant namespace
- Row-level filtering by tenant ID in metadata
- Per-tenant cost tracking and quotas
- Audit logging of every query

### 3.5 Cost estimation (cloud, 1000 queries/day)

| Component | Est. monthly cost |
|---|---|
| API (2 × 0.5 vCPU, 1 GB) | $15–25 |
| Vector store (Qdrant Cloud, 1 GB) | $25 |
| S3 (data + artifacts, 5 GB) | $1 |
| OpenRouter (10k queries, GPT-4o-mini) | $5–15 |
| **Total** | **~$50–65** |

Local model only (no hosted): ~$40–50/month.

---

## 4. Environments

| Env | Purpose | Data | Models |
|---|---|---|---|
| `dev` | Iteration, debugging | Synthetic corpus (12 docs) | Local + cached hosted |
| `ci` | Tests | Fixtures | Mocked LLM |
| `demo` | Showcase | Synthetic corpus | Local or hosted (opt-in) |
| `staging` (optional) | Pre-prod validation | Anonymized real corpus | Hosted, budget-capped |
| `prod` (out of scope v1) | — | — | — |

Env selection via `ENV=dev|ci|demo` env var. Loads `configs/{ENV}_*.yaml` if
present, else defaults.

---

## 5. Backup & restore

### 5.1 What to back up

| Artifact | Frequency | Notes |
|---|---|---|
| `data/raw/` | Once | Source PDFs; if lost, re-ingest |
| `data/eval/golden.jsonl` | On change | Version-controlled in git |
| `configs/` | On change | Version-controlled in git |
| `results/runs/*.json` | Weekly | Experiment history; regenerable but expensive |
| `logs/cost.jsonl` | Monthly | Audit trail |

### 5.2 What NOT to back up

- `data/processed/`, `data/chunks/`, `data/indexes/` — regenerable
- `data/cache/` — regenerable
- `logs/app.jsonl` — rolling, ephemeral

### 5.3 Restore

```bash
git clone <repo>
# Restore data/raw/ and results/ from backup
make install
make ingest chunk embed index
make eval-matrix    # optional; re-runs the comparison
```

---

## 6. Upgrade path (v1.x → v2)

Changes that would require a new major version:
- Vector store swap (Chroma → Qdrant)
- Judge model change (would invalidate cross-run comparisons)
- Config schema breaking change (would require migration of saved configs)

Each of these gets its own ADR when the time comes.

---

## 7. Deployment checklist (Phase 11 exit)

- [ ] `docker compose up -d` from clean clone reaches working demo
- [ ] Both services pass healthchecks within 60 s
- [ ] `curl localhost:8000/health` returns 200
- [ ] Streamlit at :8501 renders all 3 tabs
- [ ] `README.md` quick start is accurate (tested by a fresh `git clone`)
- [ ] `.env.example` lists all required vars with empty values
- [ ] Docker image < 3 GB (else slim it)
- [ ] No secrets in image layers (`docker history` check)