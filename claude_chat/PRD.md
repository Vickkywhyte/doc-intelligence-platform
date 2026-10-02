# Product Requirements Document
## Document Intelligence Platform with Evaluation

**Version:** 1.0
**Status:** Locked for v1 build
**Owner:** (you)
**Related docs:** `02_ARCHITECTURE.md`, `03_TECH_DECISIONS.md`, `06_EVAL_SPEC.md`

---

## 1. Problem statement

Enterprises sit on thousands of PDFs — contracts, policies, handbooks, RFPs,
MSAs, DPAs. LLM-powered Q&A over these documents is now table-stakes, but the
market is flooded with demos that have **no measurement of quality**.

Buyers and internal stakeholders cannot answer the three questions that
determine whether a RAG system is trustworthy:

1. **Is the answer correct?**
2. **Did the model fabricate something not in the documents?**
3. **Did it even retrieve the right passage?**

Without answers, no RAG system gets deployed past pilot. This platform solves
the **trust gap**, not the ingestion gap.

## 2. Target users

| Persona | Primary need | Success looks like |
|---|---|---|
| **Legal ops analyst** | Ask contract/policy questions, get cited answers | Answer + citation they can verify in 10 seconds |
| **ML engineer** | Compare chunking/embedding configs before rollout | One-command matrix run; heatmap they can defend |
| **Compliance officer** | Prove the system refuses when ungrounded | Refusal correctness metric ≥ 0.90 on unanswerables |
| **Engineering manager** | See cost/quality trade-offs | Cost-vs-quality scatter; per-1k-query $ |

## 3. User stories

### US-1 — Ingest
*As an analyst, I upload a folder of PDFs and the system parses, chunks,
embeds, and indexes them, reporting per-file success/failure.*

**Acceptance:**
- Accepts PDF, DOCX, TXT, MD
- Preserves page numbers and char offsets
- Returns per-file summary: pages, chars, status
- Failed files are listed with error reason, not silently dropped

### US-2 — Ask
*As an analyst, I ask a natural-language question and receive an answer with
inline citations and a confidence indicator.*

**Acceptance:**
- Answer references retrieved chunks as `[doc_id:page:chunk_id]`
- Retrieved chunks visible in UI with scores
- Latency and cost shown per query

### US-3 — Refuse
*As a compliance officer, when the answer is not in the corpus, the system
says "I don't know based on the provided documents" rather than guessing.*

**Acceptance:**
- Explicit refusal path in prompt + parser
- Refusals rendered distinctly in UI ("I don't know" badge)
- Refusal correctness measured on 15 unanswerable golden questions

### US-4 — Evaluate
*As an ML engineer, I run one command that evaluates N chunking × M embedding
configurations against a golden dataset and produces a comparison report.*

**Acceptance:**
- `make eval-matrix` runs 3×3 = 9 configs
- Completion in < 30 min on laptop (with embedding cache warm)
- Produces Markdown + heatmap + bar chart + scatter

### US-5 — Inspect a run
*As an ML engineer, I open a past experiment's JSON and see config, metrics,
per-question results, costs, and latencies.*

**Acceptance:**
- `results/runs/{run_id}.json` is a complete `EvalResult`
- Per-question drill-down available in UI
- Config hash + git SHA present

### US-6 — Compare
*As an ML engineer, I view a dashboard with heatmaps (chunking × embedding →
metric) and a cost-vs-quality scatter.*

**Acceptance:**
- Streamlit "Eval Dashboard" tab renders all charts from `results/runs/`
- Metric selectable via dropdown
- Click a config → drill into per-question results

## 4. Functional requirements

### FR-1 Ingestion
- **FR-1.1** Accept PDF, DOCX, TXT, MD.
- **FR-1.2** Extract text with page numbers preserved.
- **FR-1.3** Store parsed output as JSON: `{doc_id, pages: [{page_number, text, char_start, char_end}], metadata}`.
- **FR-1.4** Report ingestion summary: files processed, failed, total chars.
- **FR-1.5** `doc_id = sha256(file_bytes)[:16]` — stable across machines.

### FR-2 Chunking
- **FR-2.1** ≥3 strategies: fixed-size, recursive-character, semantic.
- **FR-2.2** Each chunk carries `{chunk_id, doc_id, strategy, index, text, page, char_start, char_end, token_count, metadata}`.
- **FR-2.3** Overlap configurable per strategy.
- **FR-2.4** Strategy selected via config, not code.
- **FR-2.5** `chunk_id = f"{doc_id}:{strategy}:{index:05d}"` — sortable, traceable.

### FR-3 Embedding
- **FR-3.1** ≥3 models: OpenAI `text-embedding-3-small`, BGE-small-en-v1.5, MiniLM-L6-v2.
- **FR-3.2** Batch embedding with retry + exponential backoff.
- **FR-3.3** Content-hash cache: `hash(text + model)` → vector. Second run = $0.

### FR-4 Indexing & Retrieval
- **FR-4.1** Persist vectors + metadata in Chroma (one dir per config).
- **FR-4.2** Support top-k and MMR retrieval.
- **FR-4.3** Optional cross-encoder reranker (feature-flagged; v1.1).
- **FR-4.4** Filter by doc metadata (`doc_type=contract`, etc.).

### FR-5 Generation
- **FR-5.1** Answer uses only retrieved context; prompt enforces grounding.
- **FR-5.2** Inline citations required: `[doc_id:page:chunk_id]`.
- **FR-5.3** Explicit refusal path when context insufficient.
- **FR-5.4** Pluggable LLM client (OpenAI + Anthropic minimum).
- **FR-5.5** Determinism: temperature=0, seed=0 where supported.

### FR-6 Evaluation
- **FR-6.1** Golden dataset JSONL: `{id, question, gold_answer, gold_chunk_ids: {strategy: [...]}, type}`.
- **FR-6.2** Retrieval metrics: P@k, R@k, MRR, NDCG@k for k ∈ {1,3,5}.
- **FR-6.3** Answer metrics: semantic similarity + LLM-judge correctness (1–5).
- **FR-6.4** Hallucination: claim decomposition + per-claim entailment check.
- **FR-6.5** Faithfulness: statement-level support by retrieved context.
- **FR-6.6** Per-question results persisted for drill-down.
- **FR-6.7** Unanswerable questions evaluated on refusal correctness only.
- **FR-6.8** 95% CI via bootstrap on aggregate metrics.

### FR-7 Comparison & Reporting
- **FR-7.1** Run full matrix (3×3) with one command.
- **FR-7.2** Emit CSV + Markdown + PNG charts (heatmap, bar, scatter).
- **FR-7.3** Include cost and latency per config.

### FR-8 Interfaces
- **FR-8.1** FastAPI endpoints: `/ingest`, `/query`, `/eval/run`, `/eval/runs`, `/eval/runs/{id}`, `/eval/runs/{id}/report`, `/health`.
- **FR-8.2** Streamlit UI: Chat tab, Eval Dashboard tab, Runs History tab.

## 5. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | One-command local startup | `make up` |
| NFR-2 | Full 3×3 matrix run time | < 30 min (warm cache) |
| NFR-3 | Every LLM call cost-logged | 100% |
| NFR-4 | No PII leaves machine by default | Default = local models |
| NFR-5 | Reproducibility | Same config → same metrics (±CI) |
| NFR-6 | Test coverage | ≥ 80% overall, ≥ 95% metrics |
| NFR-7 | Cold-start clone to running demo | < 10 min |
| NFR-8 | Ingestion throughput | ≥ 10 pages/sec CPU |

## 6. Out of scope (v1)

Documented so they are not proposed mid-build:

- Multi-tenant authentication
- Real-time ingestion streaming
- Fine-tuning or training
- GraphRAG / knowledge graphs
- Multilingual documents
- Kubernetes / microservices
- Managed vector databases (Pinecone, Weaviate Cloud)
- Next.js / custom frontend
- Mobile app
- Role-based access control
- Audit trail / user action logging
- Document versioning / dedup across versions

## 7. Success metrics

Measured on the golden dataset (75 questions) after Phase 9:

| Metric | Target | Rationale |
|---|---|---|
| Best config answer accuracy | ≥ 0.85 | Judge score ≥ 4.25/5 |
| Best config hallucination rate | ≤ 0.08 | Claim-level |
| Best config P@5 | ≥ 0.80 | Top-5 retrieval |
| Best config MRR | ≥ 0.75 | First relevant chunk early |
| Refusal correctness | ≥ 0.90 | On 15 unanswerables |
| Matrix run time | < 30 min | Warm cache |
| Cost per 1k queries (best local config) | $0.00 | Fully local |
| Cost per 1k queries (best hosted config) | < $2.00 | GPT-4o-mini |

**Note:** We report these as **targets**, not promises. If Phase 9 shows we
miss, we tune, re-measure, and update the target with rationale.

## 8. Constraints & assumptions

**Constraints:**
- Solo builder, time-boxed to a portfolio scope
- Must run on a laptop CPU (no GPU assumed for local models)
- Local-first by default (privacy posture)

**Assumptions:**
- Corpus size < 50k chunks per config
- Hosted LLM pricing stays roughly stable during the build
- Chunking granularity is aligned with answer granularity (i.e., a
  correct answer fits within 1–3 chunks). If false, we note it as future work
  (span-level gold annotations).

## 9. Open questions

Tracked until resolved; each resolution becomes an ADR.

- **OQ-1** Which LLM judge for the *final* matrix run? GPT-4o-mini default;
  consider GPT-4o for the headline run at +10× cost. → **Owner: you.**
- **OQ-2** Cross-encoder reranker in v1 or defer to v1.1? → Deferred to v1.1
  unless Phase 9 shows retrieval is the bottleneck. → **Owner: you.**
- **OQ-3** Sample corpus: synthetic or real PDFs? → Synthetic via
  `scripts/00_synthesize_corpus.py` for reproducibility. → **Locked.**
- **OQ-4** Do we expose hosted embeddings through the UI by default, or
  require an opt-in toggle? → Opt-in (see ADR in `11_SECURITY_PRIVACY.md`).

## 10. Sign-off

- [ ] PRD reviewed
- [ ] ADRs 001–014 written
- [ ] Roadmap agreed
- [ ] Ready to begin Phase 1

**Status:** ✅ Locked for v1 build.