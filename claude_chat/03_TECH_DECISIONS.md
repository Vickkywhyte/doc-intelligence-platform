# Architecture Decision Records (ADRs)

**Format:** Context → Decision → Consequences → Alternatives
**Status:** All ADRs 001–014 locked for v1 unless marked "proposed".

An ADR is immutable once locked. To change a decision, add a new ADR that
supersedes the old one. Do not edit historical ADRs.

---

## ADR-001 — Python 3.11 as the implementation language

**Status:** Locked

**Context:** The RAG and evaluation ecosystem (LlamaIndex, sentence-transformers,
RAGAS, unstructured) is Python-first. We need the broadest library support
with the least glue code. Time-boxed portfolio project.

**Decision:** Python 3.11.

**Consequences:**
- ✅ Best library ecosystem for RAG + eval
- ✅ Pydantic v2, structlog, FastAPI, Streamlit all mature on 3.11
- ✅ Performance acceptable for eval workloads (I/O bound)
- ⚠️ GIL limits true parallelism; mitigated by batching + async I/O
- ⚠️ Python 3.11 not 3.12 — some wheels still catching up on 3.12

**Alternatives considered:**
- TypeScript: weaker eval tooling, no RAGAS equivalent
- Rust: too slow to prototype; overkill for portfolio scope
- Python 3.12: marginal benefit, wheel availability risk

---

## ADR-002 — Chroma as the vector store

**Status:** Locked

**Context:** Need a local, zero-ops, persistent vector store with metadata
filtering. Must support per-config isolation (9 separate indexes for the
matrix). Migration path must be a config swap, not a rewrite.

**Decision:** Chroma, local persistent client, one directory per
`{strategy}_{embedder}`.

**Consequences:**
- ✅ No external service; single pip dependency
- ✅ Metadata filtering built in
- ✅ Simple directory-based persistence → easy to gitignore, easy to reset
- ✅ Swappable behind `VectorStore` ABC
- ⚠️ Not the fastest at scale; fine for < 50k chunks
- ⚠️ Concurrency: serialize writes; parallel eval reads OK

**Alternatives:**
- FAISS: fast, but no metadata filtering — disqualifying
- Qdrant: excellent, but requires a service (docker container)
- pgvector: requires Postgres; overkill for portfolio
- LanceDB: newer, less battle-tested

**Migration path:** Implement `QdrantStore(VectorStore)` in v2; config
change only.

---

## ADR-003 — Three embedding models: OpenAI small, BGE-small, MiniLM

**Status:** Locked

**Context:** The comparison is the differentiator. We need a meaningful spread
across cost, quality, and locality. Hosted + local gives us a real trade-off
story.

**Decision:**
| Role | Model | Dim | Cost |
|---|---|---|---|
| Hosted, strong | OpenAI `text-embedding-3-small` | 1536 | $0.02 / 1M tokens |
| Local, strong | `BAAI/bge-small-en-v1.5` | 384 | $0 |
| Local, baseline | `sentence-transformers/all-MiniLM-L6-v2` | 384 | $0 |

**Consequences:**
- ✅ Real cost-vs-quality trade-off (hosted = fast + costly; local = free + slower)
- ✅ BGE-small is top-tier for its size (MTEB)
- ✅ MiniLM is a ubiquitous baseline → defensible comparison
- ⚠️ Local models are CPU-slow; batch + cache
- ⚠️ Two local models ≈ 250 MB download on first run; pre-cache in Phase 4

**Alternatives:**
- Cohere embed-v3: strong, but adds a third provider
- Voyage: excellent, less ubiquitous
- E5 family: comparable to BGE, deferred to v1.1
- OpenAI `text-embedding-3-large`: 10× cost, marginal gain for portfolio

---

## ADR-004 — Three chunking strategies: fixed, recursive, semantic

**Status:** Locked

**Context:** Chunking is the highest-leverage RAG variable. We must *show*
its impact, not assert it. Need a spectrum from dumb-fast to smart-slow.

**Decision:**
| Strategy | Config | Rationale |
|---|---|---|
| Fixed | 512 tokens, 64 overlap | Dumb baseline; predictable cost |
| Recursive | LangChain-style separators, 512, 64 overlap | Industry default |
| Semantic | Embedding-similarity valley split, max 800 | Premium; tests the hypothesis that chunking matters |

**Consequences:**
- ✅ Headline chart: chunking × embedding → accuracy
- ✅ Fixed = fast baseline; semantic = expensive but often wins
- ⚠️ Semantic chunker requires embeddings *before* indexing → two-pass
- ⚠️ Semantic chunker CPU time is significant; cache aggressively

**Alternatives:**
- Parent-child chunking: interesting, deferred to v1.1
- Agentic chunking (LLM decides boundaries): too slow/costly for portfolio
- Late chunking (Jina): research-stage, deferred

---

## ADR-005 — RAGAS + custom metrics (not RAGAS alone)

**Status:** Locked

**Context:** RAGAS is a solid library with faithfulness and answer-relevancy
metrics, but its hallucination definition is opaque and LLM-judge-based.
For a project whose differentiator *is* evaluation, opaque metrics are a
liability.

**Decision:** Use RAGAS **where it's well-defined** (faithfulness,
answer_relevancy). **Write our own** for:
- Hallucination (claim decomposition + per-claim entailment)
- Retrieval precision (exact chunk-ID matching)
- Refusal correctness (unanswerable handling)
- Answer accuracy (our own judge prompt, pinned version)

**Consequences:**
- ✅ Full control over the metric definitions that matter most
- ✅ Every metric is testable on hand-crafted cases
- ✅ Interview-defensible: we can explain *exactly* how each number was computed
- ⚠️ More code to write and test
- ⚠️ Divergence from RAGAS numbers if anyone cross-references

**Alternatives:**
- RAGAS-only: fast, but shallow and opaque
- DeepEval-only: similar issue; stricter but less flexible
- LangSmith evals: hosted, adds a dependency, less control

---

## ADR-006 — LLM-as-judge with GPT-4o-mini, temperature=0

**Status:** Locked

**Context:** Human evaluation doesn't scale. GPT-4o is 10× the cost. Local
judges are noisy and slow on CPU. Need a defensible middle ground.

**Decision:** GPT-4o-mini as the default judge, `temperature=0`, structured
JSON output. Judge prompts pinned in `evaluation/judge_prompts/` with explicit
version tags (e.g., `correctness_v1.txt`). Judge version recorded in every run.

**Consequences:**
- ✅ ~$0.10–$0.50 per full 75×9 matrix run (very affordable)
- ✅ Structured JSON → deterministic parsing
- ✅ Cached by prompt hash → repeat runs nearly free
- ⚠️ Self-preference bias if generator is also OpenAI — mitigated by:
  1. Judge prompt is task-specific (entailment/correctness, not chat)
  2. Prompt version pinned and published
  3. Local judge swap supported in config
- ⚠️ GPT-4o-mini is weaker than GPT-4o on subtle correctness — acceptable
  for portfolio; noted as a config option for the "headline" run

**Alternatives:**
- GPT-4o: better, 10× cost. Available as `judge_model` config option.
- Claude Haiku: comparable; supported as config option.
- Local Llama-3.1-8B: free, but noisy and slow; not default.

---

## ADR-007 — Config-driven experiments (YAML), not CLI args

**Status:** Locked

**Context:** We run 9+ configs per matrix. Parameters must be versioned,
diffable, and shareable. "What flags did I run?" is not an acceptable answer.

**Decision:** All experiment parameters live in YAML under `configs/`.
Canonicalized YAML → SHA-256 → `config_hash` → part of `run_id`.

**Consequences:**
- ✅ Reproducibility for free
- ✅ Git-diffable experiment definitions
- ✅ Config can be shared as a single file for peer review
- ⚠️ No ad-hoc CLI overrides — deliberate, to preserve reproducibility
- ⚠️ Adding a new parameter requires schema update in `utils/config.py`

**Alternatives:**
- CLI args: unversioned, easy to forget
- Env vars: invisible, hard to diff
- Python dicts in code: not shareable outside the repo

---

## ADR-008 — FastAPI + Streamlit (not Next.js)

**Status:** Locked

**Context:** Need an API and a UI. Team is Python-first. Time-boxed. The UI
is for demos and internal use, not a customer-facing SaaS product.

**Decision:** FastAPI for the API, Streamlit for the UI.

**Consequences:**
- ✅ Ship in days, not weeks
- ✅ Streamlit renders charts, tables, chat — all needed here
- ✅ Single language (Python) across stack
- ✅ Trivial to deploy (two containers, one compose file)
- ⚠️ Streamlit is not great for high-traffic production; fine for portfolio
- ⚠️ If this were a SaaS, we'd pick Next.js — documented as v2 migration

**Alternatives:**
- Next.js + FastAPI: better UX, 3–5× build time
- Gradio: simpler, weaker layouts for a dashboard
- Plain HTML + HTMX: more control, more work

---

## ADR-009 — Unstructured for PDF parsing

**Status:** Locked

**Context:** Contracts have tables, headers/footers, multi-column layouts,
and inconsistent formatting. `pypdf` mangles these. We need page-aware output
with reasonable layout fidelity.

**Decision:** `unstructured` library, `fast` strategy for dev, `hi_res` for
production runs. Fallback to `pdfplumber` behind the `Loader` ABC if
unstructured fails on a specific doc.

**Consequences:**
- ✅ Best open-source layout fidelity for contracts
- ✅ Preserves page numbers → required for citations
- ⚠️ Heavy dependency (torch for hi_res)
- ⚠️ `hi_res` is slow; `fast` is 5–10× faster but lower fidelity
- ⚠️ Fallback path adds a code branch — tested in Phase 2

**Alternatives:**
- PyPDF2: too lossy
- pdfplumber: good, but manual layout handling
- LlamaParse: hosted, $, adds external dependency
- Azure Document Intelligence: strong, but cloud-only

---

## ADR-010 — Golden dataset is human-curated, versioned in git

**Status:** Locked

**Context:** The entire evaluation is only as good as its ground truth. An
LLM-generated gold set is circular — the judge and the generator share blind
spots.

**Decision:** `data/eval/golden.jsonl` is version-controlled, edited via PR,
never auto-generated (except an initial synthetic seed that is then
human-reviewed). Each entry has `type: factual | multi_hop | unanswerable`.
20% of entries (15/75) are unanswerable to measure refusal correctness.

**Consequences:**
- ✅ Trustworthy ground truth
- ✅ Changes are auditable in git history
- ✅ Cross-strategy gold chunk IDs captured at curation time
- ⚠️ Slow to grow — acceptable; quality > quantity
- ⚠️ Requires discipline: no `scripts/` file modifies `golden.jsonl`

**Alternatives:**
- LLM-generated gold: circular, untrustworthy
- Crowdsourced: expensive, slow, inconsistent for legal domain
- Real user queries (production logs): not available for a new project

---

## ADR-011 — Hallucination is defined as: unsupported claims in the answer

**Status:** Locked

**Context:** "Hallucination" is used loosely in the RAG discourse. A testable
definition is required for the metric to be defensible.

**Decision:** A claim is **hallucinated** iff it appears in the answer but is
**not entailed** by any retrieved chunk. Operationalized as:

1. **Claim decomposition** (LLM, temp=0): decompose the answer into atomic
   factual claims.
2. **Entailment check** per claim (LLM, temp=0): for each claim, ask "Is this
   claim entailed by any of these chunks? yes | partial | no."
3. **Rate:**
hallucination_rate = (# claims with verdict "no")
/ (# claims)

"partial" counts as 0.5 hallucinated (configurable).

**Exclusions (documented):**
- Claims that restate the question are not counted.
- Claims in a refusal are trivially non-hallucinated.
- If an answer has zero atomic claims → rate = 0, flagged in report.

**Why not just "ask if the answer is hallucinated?"** Single-shot judgments
are opaque and inconsistent. Decomposition makes each verdict inspectable,
cacheable, and testable.

**Consequences:**
- ✅ Auditable per-claim
- ✅ Unit-testable on hand-crafted cases
- ✅ Cacheable by claim hash
- ⚠️ Slower than single-shot: ~1 extra LLM call per claim
- ⚠️ Claim decomposition quality varies; logged

**Alternatives:**
- Single-shot hallucination judge: opaque, unreliable
- NLI-only (no LLM): requires a strong NLI model; ambiguous on legal text
- Token overlap heuristics: too brittle

---

## ADR-012 — Faithfulness ≠ Hallucination

**Status:** Locked

**Context:** These two are often conflated. They measure different failure
modes and answer different stakeholder questions.

**Decision:**
- **Hallucination** = claims in the *answer* not entailed by retrieved
context. Answers: *"Did the model make something up?"*
- **Faithfulness** = fraction of the answer supported by retrieved context.
Answers: *"Is the answer grounded in what was retrieved?"*

Both are reported separately. A model can be faithful (uses context) but
hallucinatory (adds extras), or vice versa.

**Operationalization of faithfulness** (RAGAS-style):
faithfulness = (# statements in answer supported by retrieved context)
/ (# statements in answer)
Looser than hallucination: allows paraphrase/summarization as long as
grounded.

**Consequences:**
- ✅ Two columns in every report → clearer story
- ✅ Different diagnostics: hallucination → prompt fix; low faithfulness →
  retrieval fix
- ⚠️ Overlap in computation; can share intermediate artifacts (claim
  decomposition)
- ⚠️ Report readers may conflate them → labeled clearly in every chart

**Alternatives:**
- Merge into a single "groundedness" score: loses signal, hurts diagnostics

---

## ADR-013 — Refusal is a first-class outcome

**Status:** Locked

**Context:** Most RAG systems are graded only on answers. Silence (correct
refusal) is a feature, not a bug. Without measuring it, we can't prove the
system is safe.

**Decision:** Golden dataset includes `type: unanswerable` questions (15/75 =
20%). The pipeline has an explicit refusal path (prompt instruction + parser
detection). Metrics include `refusal_correctness`:
refusal_correctness = (# unanswerable where response.refused)
/ (# unanswerable)


**Consequences:**
- ✅ Trust signal measurable and reportable
- ✅ Forces prompt discipline (system must know what it doesn't know)
- ✅ Refusals rendered distinctly in UI
- ⚠️ Over-refusal is a failure mode too — tracked via accuracy on answerables
- ⚠️ Detection is regex + LLM-judge hybrid to avoid false positives

**Alternatives:**
- Ignore unanswerables: inflates accuracy, hides failure mode
- Force answers always: unsafe, unusable in legal/compliance

---

## ADR-014 — Streamlit app has an Eval Dashboard tab

**Status:** Locked

**Context:** Evaluation results in a Markdown file are easy to ignore.
The project's differentiator must be visible in the first 30 seconds of a
demo. A dashboard that renders the comparison *is* the pitch.

**Decision:** Streamlit app has three tabs:
1. **Chat** — Q&A with citations
2. **Eval Dashboard** — heatmap (chunking × embedding → metric), hallucination
   bar chart, cost-vs-quality scatter, full results table
3. **Runs History** — table of past runs with drill-down

All charts read from `results/runs/*.json`. No hardcoded data.

**Consequences:**
- ✅ The demo *is* the comparison
- ✅ Metric selectable via dropdown → flexible story
- ✅ Drill-down from chart to per-question → trustworthy detail
- ⚠️ Streamlit performance with large runs; paginate per-question table
- ⚠️ Plotly dependency added (`plotly>=5.20`)

**Alternatives:**
- Notebooks only: not demoable in 30 seconds
- Tableau/Metabase: overkill for portfolio; adds infrastructure
- Static HTML report: not interactive; can't drill down

---

## ADR-015 — Two local embedders instead of three (supersedes ADR-003 scope)

**Status:** Locked

**Context:** OpenRouter does not proxy embedding models. Verified against
their /v1/models endpoint — openai/text-embedding-3-small and
openai/text-embedding-3-large are absent. ADR-003 assumed hosted embeddings
would be accessible through the same gateway used for generation. Adding a
direct OpenAI key was ruled out for cost reasons.

**Decision:** Reduce the embedding matrix from three models to two, both
local:
- BAAI/bge-small-en-v1.5 (384d, top MTEB per size)
- sentence-transformers/all-MiniLM-L6-v2 (384d, ubiquitous baseline)

Hosted generation and judging still go through OpenRouter. The evaluation
matrix becomes 3 chunking × 2 embedding = 6 configs (down from 9).

**Consequences:**
- Cost-vs-quality trade-off is narrower (no hosted embedder to compare against)
- All 6 configs run with $0 embedding cost
- Latency comparison is still meaningful (BGE vs MiniLM have different speeds)
- If a hosted embedder is added in v2, it goes behind the existing Embedder ABC
  with no other changes needed
- ADR-003's Chroma, registry, and pipeline decisions stand unchanged

**Alternatives:**
- Direct OpenAI key for embeddings: rejected on cost constraint
- Cohere / Voyage hosted embedder via OpenRouter: not supported (same limitation)
- Skip hosted entirely and lose the trade-off story: rejected — 2 local embedders
  still produce a meaningful comparison

---

## ADR index

| ID | Title | Status |
|---|---|---|
| ADR-001 | Python 3.11 | Locked |
| ADR-002 | Chroma vector store | Locked |
| ADR-003 | Three embedding models | Locked |
| ADR-004 | Three chunking strategies | Locked |
| ADR-005 | RAGAS + custom metrics | Locked |
| ADR-006 | GPT-4o-mini judge | Locked |
| ADR-007 | YAML config-driven | Locked |
| ADR-008 | FastAPI + Streamlit | Locked |
| ADR-009 | Unstructured PDF parsing | Locked |
| ADR-010 | Human-curated golden set | Locked |
| ADR-011 | Hallucination = unsupported claims | Locked |
| ADR-012 | Faithfulness ≠ Hallucination | Locked |
| ADR-013 | Refusal is first-class | Locked |
| ADR-014 | Eval Dashboard tab | Locked |
| ADR-015 | Two local embedders (supersedes ADR-003 scope) | Locked |

**How to amend:** Add a new ADR (e.g., ADR-016) titled
"Supersedes ADR-XXX — ..." and update this index. Never edit a locked ADR
in place.