# Architecture

**Related docs:** `01_PRD.md`, `03_TECH_DECISIONS.md`, `05_DATA_MODEL.md`,
`06_EVAL_SPEC.md`

---

## 1. System overview
┌─────────────────────────────────────────────────────────────────────┐
│ INTERFACES │
│ │
│ Streamlit UI │ FastAPI │ CLI scripts │ Notebooks │
│ (chat, eval, │ (/query, │ (01–06_.py) │ (exploration) │
│ runs tabs) │ /eval) │ │ │
└─────────┬────────┴──────┬──────┴────────┬────────┴─────────┬─────────┘
│ │ │ │
▼ ▼ ▼ ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATION LAYER │
│ │
│ rag_pipeline.py experiment.py evaluator.py │
│ (Q → A + citations) (matrix runner) (metrics + judge) │
└─────┬──────────────┬────────────────┬──────────────────┬─────────────┘
│ │ │ │
▼ ▼ ▼ ▼
┌──────────┐ ┌───────────┐ ┌────────────┐ ┌────────────────┐
│Ingestion │ │ Chunking │ │ Embeddings │ │ Evaluation │
│ │ │ │ │ │ │ │
│ loader │ │ fixed │ │ openai │ │ metrics/ │
│ cleaner │ │ recursive │ │ bge │ │ judge │
│ metadata │ │ semantic │ │ minilm │ │ report │
└────┬─────┘ └─────┬─────┘ └─────┬──────┘ └────────┬───────┘
│ │ │ │
▼ ▼ ▼ ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PERSISTENCE / INFRA LAYER │
│ │
│ FileSystem (data/) │ Chroma │ SQLite (run registry) │
│ Cost log (JSONL) │ Cache │ (embeddings, judge verdicts) │
└─────────────────────────────────────────────────────────────────────┘


## 2. Component responsibilities

| Component | Responsibility | Key interface |
|---|---|---|
| `ingestion` | PDF/DOCX → normalized text + metadata | `Loader.load(path) -> Document` |
| `chunking` | Document → List[Chunk] | `ChunkStrategy.chunk(doc) -> list[Chunk]` |
| `embeddings` | Text → vector | `Embedder.embed(texts) -> np.ndarray` |
| `vectorstore` | Persist & query vectors + metadata | `VectorStore.upsert/query` |
| `retrieval` | Query → ranked chunks | `Retriever.retrieve(q, k) -> list[RetrievedChunk]` |
| `generation` | Chunks + Q → answer + citations | `LLMClient.generate(prompt) -> LLMResponse` |
| `pipeline` | Wire everything | `RAGPipeline.query(q) -> RAGResponse` |
| `evaluation` | (Q, answer, chunks, gold) → metrics | `Evaluator.run(config) -> EvalResult` |
| `api` | HTTP surface | FastAPI routers |
| `ui` | Human surface | Streamlit app |

Full interfaces (ABCs, Pydantic models) in `05_DATA_MODEL.md`.

## 3. Data flows

### 3.1 Ingestion → Index (offline, one-time per config)
PDF/DOCX/TXT
│
▼
┌──────────┐
│ Loader │ (Unstructured, fallback pdfplumber)
└────┬─────┘
│ Document{text, pages, metadata}
▼
┌──────────┐
│ Cleaner │ (whitespace, dehyphenation, header/footer strip)
└────┬─────┘
│ Document (clean)
▼
┌──────────┐
│ Chunker │ (selected by config: fixed | recursive | semantic)
└────┬─────┘
│ List[Chunk] → data/chunks/{strategy}/{doc_id}.jsonl
▼
┌──────────┐
│ Embedder │ (batched, cached by hash(text+model))
└────┬─────┘
│ np.ndarray[N, dim]
▼
┌─────────────┐
│ VectorStore │ (Chroma, one dir per {strategy}_{embedder})
└─────────────┘

### 3.2 Query → Answer (online, per user question)
User Question
│
▼
┌──────────┐
│ Embedder │
└────┬─────┘
│ query vector
▼
┌─────────────┐
│ VectorStore │ top-k + optional metadata filter
└────┬────────┘
│ List[RetrievedChunk]
▼
┌──────────┐
│ Reranker │ (optional, feature-flagged)
└────┬─────┘
│ List[RetrievedChunk] reordered
▼
┌────────────────┐
│ PromptTemplate │ (system + context + question, refusal instruction)
└────┬───────────┘
│ prompt string
▼
┌───────────┐
│ LLMClient │ (OpenAI | Anthropic)
└────┬──────┘
│ raw text
▼
┌────────────────┐
│ CitationParser │ extract [doc:page:chunk] + refusal detection
└────┬───────────┘
│
▼
RAGResponse{answer, refused, citations, retrieved, model,
latency_ms, cost_usd}

### 3.3 Evaluation (offline, per matrix run)
Golden Dataset (75 Q&A, JSONL)
│
▼
For each config in matrix (3 chunking × 3 embedding = 9):
│
├─► For each question:
│ ├── Retrieve top-k
│ ├── Generate answer
│ ├── Compute retrieval metrics (vs gold_chunk_ids[strategy])
│ ├── Compute answer metrics (vs gold_answer)
│ ├── Decompose answer → claims → entailment → hallucination
│ ├── Compute faithfulness (statement-level support)
│ └── Persist QuestionResult
│
▼
EvalResult ──► results/runs/{run_id}.json
│
▼
report.py ──► comparison.md
heatmap.png
hallucination_bar.png
cost_quality_scatter.png

## 4. Storage layout

| Path | Contents | Gitignored? |
|---|---|---|
| `data/raw/` | Source PDFs (immutable) | ✅ |
| `data/processed/{doc_id}.json` | Parsed text + page offsets | ✅ |
| `data/chunks/{strategy}/{doc_id}.jsonl` | Chunks per strategy | ✅ |
| `data/indexes/{strategy}_{embedder}/` | Chroma persistence | ✅ |
| `data/eval/golden.jsonl` | Human-curated Q&A | ❌ (versioned) |
| `data/cache/embeddings/{hash}.npy` | Content-hash → vector | ✅ |
| `data/cache/judge/{hash}.json` | Prompt-hash → verdict | ✅ |
| `results/runs/{run_id}.json` | Full EvalResult | ✅ |
| `results/reports/{run_id}.md` | Human-readable report | ✅ |
| `results/figures/{run_id}_*.png` | Charts | ✅ |
| `logs/app.jsonl` | Structured application logs | ✅ |
| `logs/cost.jsonl` | Append-only cost ledger | ✅ |
| `data/registry.sqlite` | Run registry (metadata index) | ✅ |

## 5. Configuration model

Every experiment is fully described by one YAML. Canonicalized (sorted keys,
no whitespace) → SHA-256 → `config_hash` → part of `run_id`.

```yaml
# configs/experiment_matrix.yaml
chunking:
  - {name: fixed,     size: 512, overlap: 64}
  - {name: recursive, size: 512, overlap: 64}
  - {name: semantic,  threshold: 0.75, max_size: 800}

embeddings:
  - {name: openai_text_embedding_3_small, batch: 100}
  - {name: bge_small_en_v15,              batch: 64}
  - {name: minilm_l6_v2,                  batch: 64}

retrieval:
  k: 5
  mmr: false
  rerank: false

generation:
  model: gpt-4o-mini
  temperature: 0
  max_tokens: 512

evaluation:
  judge_model: gpt-4o-mini
  judge_prompt_version: v1
  metrics: [p_at_k, recall_at_k, mrr, ndcg, answer_accuracy,
            hallucination, faithfulness, refusal_correctness]
  k_values: [1, 3, 5]
  bootstrap_samples: 1000
  budget_usd: 5.00
  seed: 0
6. Extension points
Add a...	Do this
New chunker	Subclass ChunkStrategy in chunking/, register in chunking/registry.py, add config option
New embedder	Subclass Embedder in embeddings/, register in embeddings/registry.py
New vector store	Subclass VectorStore in vectorstore/, register in vectorstore/registry.py
New retriever	Subclass Retriever in retrieval/, wire in retrieval/registry.py
New LLM provider	Subclass LLMClient in generation/, register in llm_client.py
New metric	Subclass Metric in evaluation/metrics/, add to 06_EVAL_SPEC.md, add test in tests/test_metrics.py
New judge prompt	Add file to evaluation/judge_prompts/{name}_v{N}.txt, reference in config
Extension points exist so we never need to modify call sites to add
implementations. Enforced by registry pattern.
7. Cross-cutting concerns
7.1 Logging
structlog with JSON output

Every log line has: ts, level, event, run_id (if in a run),
doc_id/chunk_id (if applicable), latency_ms, cost_usd

Destinations: stdout (dev) + logs/app.jsonl (rotating)

No PII in logs. Documents referenced by doc_id, not filename when
sensitive.

7.2 Cost tracking
utils/cost_tracker.py maintains process-local counter + appends to
logs/cost.jsonl

Price table in configs/prices.yaml (dated, easy to update)

Hard cap: if total_cost_usd > config.evaluation.budget_usd, run
aborts with clear error before exceeding

7.3 Caching
Two caches, both content-hash keyed:

Embeddings: hash(text + model_name) → .npy vector

Judge: hash(prompt + model_name + prompt_version) → JSON verdict

Disable via --no-cache for benchmark purity (rarely needed).

7.4 Determinism
random.seed(config.evaluation.seed)

np.random.seed(config.evaluation.seed)

LLM calls: temperature=0, provider seed= where supported

Nondeterminism sources are logged (e.g., "provider does not support seed")
7.5 Reproducibility
Every run records:

config_hash, git_sha, python_version

package_versions (from importlib.metadata)

judge_model, judge_prompt_version

seed, started_at, ended_at

8. Deployment topology (v1)
text
┌────────────────────────────────────────────────┐
│  Docker Compose                                │
│                                                │
│  ┌──────────┐         ┌──────────┐             │
│  │   api    │◄────────│    ui    │             │
│  │  :8000   │         │  :8501   │             │
│  └────┬─────┘         └────┬─────┘             │
│       │                    │                   │
│       └────────┬───────────┘                   │
│                │                               │
│         ┌──────▼──────┐                        │
│         │  volumes:   │                        │
│         │  ./data      │                        │
│         │  ./results   │                        │
│         │  ./logs      │                        │
│         └─────────────┘                        │
│                                                │
└────────────────────────────────────────────────┘
Monolith with clean module boundaries. No microservices. No Kubernetes.

9. Explicit non-choices (v1)
Documented so they are not proposed:

No microservices. Monolith, importable package.

No Kubernetes. Docker Compose only.

No managed vector DB. Chroma local; path is config.

No message queue. Scripts and API are synchronous.

No auth. Local-only default; auth documented as a cloud-path addition.

No GPU assumption. Local models run on CPU; slow but functional.

10. Design principles (recap)
Interfaces over implementations. ABCs + registries.

Config over code. YAML drives everything.

Reproducibility is a feature. Config hash + git SHA on every run.

Cost-aware by default. Logged, cached, capped.

Fail loud in dev, soft in prod. Typed exceptions; per-doc skips logged.

Evaluate before optimizing. No change without baseline metric.