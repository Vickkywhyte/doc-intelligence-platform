# Data Model & Interfaces

**Related docs:** `02_ARCHITECTURE.md`, `06_EVAL_SPEC.md`, `07_API_SPEC.md`

Source of truth for every Pydantic model and ABC in the system.

---

## 1. Conventions

- **Pydantic v2** for all models crossing module boundaries.
- **`extra="forbid"`** on all config models (catch typos early).
- **`extra="ignore"`** on all I/O models (forward compatibility).
- All timestamps are timezone-aware UTC (`datetime.now(timezone.utc)`).
- All IDs are strings (never ints — chunk IDs are hex-ish).
- Optional fields use `T | None`, never `Optional[T]`.

---

## 2. Core models

### 2.1 Document

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal

DocType = Literal["contract", "policy", "handbook", "rfp", "dpa", "nda", "other"]

class Page(BaseModel):
    model_config = {"extra": "ignore"}

    page_number: int = Field(ge=1)
    text: str
    char_start: int = Field(ge=0)   # offset into concatenated doc text
    char_end: int = Field(ge=0)

class Document(BaseModel):
    model_config = {"extra": "ignore"}

    doc_id: str                     # sha256(file_bytes)[:16]
    source_path: str
    filename: str
    doc_type: DocType = "other"
    pages: list[Page]
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime

    @property
    def full_text(self) -> str:
        return "".join(p.text for p in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)
```

### 2.2 Chunk

```python
class Chunk(BaseModel):
    model_config = {"extra": "ignore"}

    chunk_id: str                   # f"{doc_id}:{strategy}:{index:05d}"
    doc_id: str
    strategy: str                   # "fixed" | "recursive" | "semantic"
    index: int = Field(ge=0)
    text: str
    page: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 2.3 Retrieval

```python
class RetrievedChunk(BaseModel):
    model_config = {"extra": "ignore"}

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: Literal["topk", "mmr", "rerank"]
```

### 2.4 Generation

```python
class Citation(BaseModel):
    model_config = {"extra": "ignore"}

    doc_id: str
    page: int
    chunk_id: str

class LLMResponse(BaseModel):
    model_config = {"extra": "ignore"}

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int

class RAGResponse(BaseModel):
    model_config = {"extra": "ignore"}

    query: str
    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    model: str
    latency_ms: int
    cost_usd: float
    warnings: list[str] = Field(default_factory=list)
```

### 2.5 Evaluation

```python
EvalType = Literal["factual", "multi_hop", "unanswerable"]

class GoldenItem(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    question: str
    gold_answer: str
    gold_chunk_ids: dict[str, list[str]]   # {strategy: [chunk_id, ...]}
    type: EvalType
    notes: str | None = None

class QuestionResult(BaseModel):
    model_config = {"extra": "ignore"}

    item_id: str
    question: str
    answer: str
    refused: bool
    retrieved_chunk_ids: list[str]
    metrics: dict[str, float]
    hallucinated_claims: list[str]
    judge_verdict: dict[str, Any]
    cost_usd: float
    latency_ms: int

class EvalResult(BaseModel):
    model_config = {"extra": "ignore"}

    run_id: str                     # f"{YYYYMMDDTHHMMSS}_{config_hash[:8]}"
    config_hash: str
    config: dict[str, Any]
    git_sha: str | None
    python_version: str
    package_versions: dict[str, str]
    judge_model: str
    judge_prompt_version: str
    seed: int
    started_at: datetime
    ended_at: datetime
    n_questions: int
    aggregate_metrics: dict[str, float]
    aggregate_ci: dict[str, tuple[float, float]]   # 95% bootstrap
    per_question: list[QuestionResult]
    total_cost_usd: float
    notes: str | None = None
```

---

## 3. Abstract base classes (interfaces)

### 3.1 ChunkStrategy

```python
from abc import ABC, abstractmethod

class ChunkStrategy(ABC):
    """A chunking strategy. Stateless; config passed at construction."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Return chunks in reading order with valid offsets."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'recursive'."""
        ...
```

### 3.2 Embedder

```python
import numpy as np

class Embedder(ABC):
    """Text → dense vectors. Handles batching + caching internally."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape [len(texts), dim], float32, L2-normalized."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'bge_small_en_v15'."""
        ...
```

### 3.3 VectorStore

```python
class VectorStore(ABC):
    """Persist + query vectors with metadata filtering."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def query(
        self,
        vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
```

### 3.4 Retriever

```python
class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
```

### 3.5 LLMClient

```python
class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
```

### 3.6 Metric

```python
class Metric(ABC):
    """Computes 1+ numeric scores for one (GoldenItem, RAGResponse) pair."""

    @abstractmethod
    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        """Return {metric_name: value}. Names are flat, e.g. 'p_at_5'."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
```

---

## 4. On-disk formats

| Artifact | Format | Path | Notes |
|---|---|---|---|
| Raw source | PDF/DOCX/TXT/MD | `data/raw/{filename}` | Immutable |
| Parsed doc | JSON (one file per doc) | `data/processed/{doc_id}.json` | `Document` model |
| Chunks | JSONL (one line per chunk) | `data/chunks/{strategy}/{doc_id}.jsonl` | `Chunk` model |
| Embedding cache | `.npy` + `.json` sidecar | `data/cache/embeddings/{hash}.npy` | `{hash}.json` = `{model, dim, text_preview}` |
| Judge cache | JSON | `data/cache/judge/{hash}.json` | `{model, prompt_version, verdict}` |
| Chroma index | Directory | `data/indexes/{strategy}_{embedder}/` | Persistent client |
| Golden set | JSONL | `data/eval/golden.jsonl` | Version-controlled |
| Run result | JSON | `results/runs/{run_id}.json` | `EvalResult` model |
| Report | Markdown | `results/reports/{run_id}.md` | Human-readable |
| CSV | CSV | `results/reports/{run_id}.csv` | For downstream analysis |
| Figures | PNG | `results/figures/{run_id}_*.png` | Plotly → kaleido |
| App log | JSONL | `logs/app.jsonl` | structlog output |
| Cost log | JSONL (append-only) | `logs/cost.jsonl` | `{ts, model, tokens, cost_usd, run_id}` |

---

## 5. ID conventions

| ID | Format | Example | Property |
|---|---|---|---|
| `doc_id` | `sha256(file_bytes)[:16]` | `a1b2c3d4e5f6a7b8` | Stable across machines |
| `chunk_id` | `{doc_id}:{strategy}:{index:05d}` | `a1b2...:recursive:00012` | Sortable, traceable |
| `run_id` | `{YYYYMMDDTHHMMSS}_{config_hash[:8]}` | `20250115T103000_a1b2c3d4` | Time-ordered, unique |
| `config_hash` | `sha256(canonical_yaml)` | `a1b2c3d4...` (full 64) | Deterministic |
| `citation` | `[doc_id:page:chunk_id]` | `[a1b2...:7:a1b2...:recursive:00012]` | Rendered in answer |

---

## 6. Config schema (Pydantic)

```python
class ChunkingConfig(BaseModel):
    model_config = {"extra": "forbid"}
    name: Literal["fixed", "recursive", "semantic"]
    size: int = 512
    overlap: int = 64
    threshold: float | None = None    # semantic only
    max_size: int | None = None       # semantic only

class EmbeddingConfig(BaseModel):
    model_config = {"extra": "forbid"}
    name: Literal["openai_text_embedding_3_small", "bge_small_en_v15", "minilm_l6_v2"]
    batch: int = 64

class RetrievalConfig(BaseModel):
    model_config = {"extra": "forbid"}
    k: int = 5
    mmr: bool = False
    mmr_lambda: float = 0.5
    rerank: bool = False

class GenerationConfig(BaseModel):
    model_config = {"extra": "forbid"}
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512

class EvaluationConfig(BaseModel):
    model_config = {"extra": "forbid"}
    judge_model: str = "gpt-4o-mini"
    judge_prompt_version: str = "v1"
    metrics: list[str]
    k_values: list[int] = [1, 3, 5]
    bootstrap_samples: int = 1000
    budget_usd: float = 5.00
    seed: int = 0
    partial_claim_weight: float = 0.5    # for hallucination

class ExperimentConfig(BaseModel):
    model_config = {"extra": "forbid"}
    chunking: list[ChunkingConfig]
    embeddings: list[EmbeddingConfig]
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    def hash(self) -> str:
        """SHA-256 of canonicalized YAML (sorted keys, no whitespace)."""
        ...
```

---

## 7. Full canonical config file

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
  mmr_lambda: 0.5
  rerank: false

generation:
  model: openai/gpt-4o-mini          # OpenRouter model string
  temperature: 0.0
  max_tokens: 512

evaluation:
  judge_model: openai/gpt-4o-mini    # OpenRouter model string
  judge_prompt_version: v1
  metrics:
    - p_at_k
    - recall_at_k
    - mrr
    - ndcg
    - answer_accuracy
    - hallucination
    - faithfulness
    - refusal_correctness
  k_values: [1, 3, 5]
  bootstrap_samples: 1000
  budget_usd: 5.00
  seed: 0
  partial_claim_weight: 0.5
```

---

## 8. Response model examples

### 8.1 `/query` response
```json
{
  "query": "What is the notice period for termination without cause?",
  "answer": "30 days written notice [a1b2c3d4e5f6a7b8:7:a1b2c3d4e5f6a7b8:recursive:00012].",
  "refused": false,
  "citations": [
    {
      "doc_id": "a1b2c3d4e5f6a7b8",
      "page": 7,
      "chunk_id": "a1b2c3d4e5f6a7b8:recursive:00012"
    }
  ],
  "retrieved": [
    {
      "chunk": {
        "chunk_id": "a1b2c3d4e5f6a7b8:recursive:00012",
        "doc_id": "a1b2c3d4e5f6a7b8",
        "strategy": "recursive",
        "index": 12,
        "text": "Either party may terminate this Agreement without cause upon thirty (30) days' prior written notice...",
        "page": 7,
        "char_start": 4210,
        "char_end": 4588,
        "token_count": 78,
        "metadata": {"section": "7.2"}
      },
      "score": 0.83,
      "rank": 1,
      "retriever": "topk"
    }
  ],
  "model": "openai/gpt-4o-mini",
  "latency_ms": 1240,
  "cost_usd": 0.0004,
  "warnings": []
}
```

### 8.2 `EvalResult` (abbreviated)
```json
{
  "run_id": "20250115T103000_a1b2c3d4",
  "config_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
  "config": { ... full experiment config ... },
  "git_sha": "f3a9b21...",
  "python_version": "3.11.9",
  "package_versions": {"pydantic": "2.8.2", "chromadb": "0.5.5"},
  "judge_model": "openai/gpt-4o-mini",
  "judge_prompt_version": "v1",
  "seed": 0,
  "started_at": "2025-01-15T10:30:00Z",
  "ended_at":   "2025-01-15T10:47:12Z",
  "n_questions": 75,
  "aggregate_metrics": {
    "p_at_5": 0.83,
    "mrr": 0.77,
    "ndcg_at_5": 0.81,
    "answer_accuracy": 0.87,
    "sem_sim": 0.79,
    "hallucination_rate": 0.05,
    "faithfulness": 0.91,
    "refusal_correctness": 0.93,
    "latency_ms_p50": 1320,
    "cost_per_1k_q_usd": 0.42
  },
  "aggregate_ci": {
    "answer_accuracy": [0.83, 0.91],
    "hallucination_rate": [0.03, 0.08]
  },
  "per_question": [ ... 75 QuestionResult objects ... ],
  "total_cost_usd": 0.031,
  "notes": null
}
```

---

## 9. Cache key specifications

| Cache | Key | Value | Invalidation |
|---|---|---|---|
| Embeddings | `sha256(text + "||" + model_name)` | `.npy` + sidecar `.json` | Manual (`make clean-cache`) |
| Judge | `sha256(prompt + "||" + model + "||" + prompt_version)` | JSON verdict | Bump `judge_prompt_version` |
| Retrieval | None (always fresh; cheap) | — | — |
| Generation | None (expensive but rare; correctness over speed) | — | — |

**Note:** Generation is deliberately **not** cached — the LLM response is the
artifact we're evaluating, and caching across runs could hide provider drift.
Judge verdicts *are* cached because we've frozen the prompt version.

---

## 10. Schema versioning

The `EvalResult` schema is versioned implicitly by `package_versions`.
If a future change to the schema would break parsing, add a `schema_version`
field (default `"1"`) and handle migration in `evaluation/report.py`.

For v1: no explicit schema version; the docs in this folder are the contract.