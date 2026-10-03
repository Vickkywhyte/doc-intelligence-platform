# Document Intelligence Platform — Start Here

> **TL;DR:** A RAG system that ingests company documents, answers questions
> with citations, and — the actual point — measures its own quality with a
> first-class evaluation pipeline comparing chunking strategies and embedding
> models.

---

## What this is

A production-grade **Document Intelligence Platform** with a **first-class
evaluation pipeline**. Users upload PDFs (contracts, policies, handbooks,
RFPs), ask natural-language questions, and receive cited, grounded answers.
Simultaneously, an automated harness measures every answer against a
human-curated golden dataset across a matrix of 3 chunking strategies × 3
embedding models.

## Why it exists

Most RAG projects are demos: a PDF upload box, a chat window, no measurement.
Stakeholders can't answer the three questions that matter:

1. **Is the answer correct?** (accuracy)
2. **Did the model make something up?** (hallucination)
3. **Did it even retrieve the right passage?** (retrieval precision)

This platform solves the **trust gap**, not the ingestion gap. The deliverable
is a benchmark harness with a chat UI on top — not a chatbot demo.

## Who it's for

| Persona | Their need |
|---|---|
| Legal ops analyst | Ask contract/policy questions, get cited answers |
| ML engineer | Compare chunking/embedding configs before rollout |
| Compliance officer | Prove the system refuses when ungrounded |
| Engineering manager | See cost/quality trade-offs in one dashboard |

## The differentiator, in one sentence

**Everyone builds RAG; almost nobody evaluates it properly.** This project
treats evaluation as a product feature with precise, testable metric
definitions — hallucination and faithfulness are defined separately, refusal
is a first-class outcome, and every run is reproducible from a config hash.

---

## Reading order

Read in this sequence. Each document assumes the previous ones.

### Tier 1 — Foundations (why + what)
1. **`PROJECT_INSTRUCTIONS.md`** — how to work in this repo
2. **`CLAUDE.md`** — operating rules for Claude Code / AI assistance
3. **`01_PRD.md`** — what we're building and why
4. **`02_ARCHITECTURE.md`** — how it fits together
5. **`03_TECH_DECISIONS.md`** — the 14 ADRs behind every choice
6. **`04_ROADMAP.md`** — 11 phases, in build order

### Tier 2 — Execution (how)
7. **`05_DATA_MODEL.md`** — Pydantic models, on-disk formats, ABC contracts
8. **`06_EVAL_SPEC.md`** — ⭐ the differentiator, read carefully
9. **`07_API_SPEC.md`** — REST contract
10. **`08_UI_SPEC.md`** — Streamlit app spec
11. **`09_TESTING_STRATEGY.md`** — test pyramid and coverage targets
12. **`10_OBSERVABILITY.md`** — logging, cost tracking, run artifacts
13. **`11_SECURITY_PRIVACY.md`** — data handling and hosted-model opt-in

### Tier 3 — Reference
14. **`12_DEPLOYMENT.md`** — local, Docker Compose, cloud path
15. **`13_RISKS_AND_MITIGATIONS.md`** — 12 identified risks
16. **`14_GLOSSARY.md`** — RAG/eval terminology
17. **`CHANGELOG.md`** — running log of decisions

---

## Current status

| Phase | Status | Notes |
|---|---|---|
| Planning (`claude_chat/`) | 🟢 Complete | 19 docs |
| Phases 1–11 | 🟢 Complete | All shipped |
| **PROJECT STATUS** | **🟢 CLOSED** | **Live on GitHub** |

**Repo:** https://github.com/Vickkywhyte/doc-intelligence-platform
**CI:** ✅ passing
**Tests:** 378 passing locally · 376 + 2 skipped in CI
**Cost to date:** ~$0.77 total LLM spend

## Final summary

| Metric | Value |
|---|---|
| Phases completed | 11 / 11 |
| Test suite | 378 passing |
| ADRs locked | 15 |
| Golden dataset | 75 unique questions (45 factual / 15 multi_hop / 15 unanswerable) |
| Matrix configs | 6 (3 chunkers × 2 embedders) |
| Best accuracy | sem·mini 0.933 |
| Best MRR | sem·bge 0.797 |
| Lowest hallucination | rec·mini 0.031 |
| Cheapest config | fix·bge $0.80 / 1k queries |
| Total OpenRouter spend | ~$0.77 |
| Docker | Multi-stage, one-command startup |
| CI | Green on GitHub Actions |
| Documentation | 19 planning docs + README + CONTRIBUTING |

---

## Project Definition of Done

The project is complete when **all** of these hold:

### Functional
- [ ] Ingests PDF, DOCX, TXT, MD with page-aware offsets
- [ ] 3 chunking strategies (fixed, recursive, semantic) behind one ABC
- [ ] 3 embedding models (OpenAI small, BGE-small, MiniLM) behind one ABC
- [ ] Retrieval with top-k + MMR + optional reranker
- [ ] Grounded generation with inline `[doc:page:chunk]` citations
- [ ] Explicit refusal path for unanswerable questions
- [ ] FastAPI with `/ingest`, `/query`, `/eval/run`, `/eval/runs/{id}`
- [ ] Streamlit UI with Chat, Eval Dashboard, and Runs tabs

### Evaluation
- [ ] Golden dataset ≥ 75 Q&A pairs, human-verified, PR-reviewed
  - 45 factual · 15 multi-hop · 15 unanswerable
- [ ] Metrics implemented + unit-tested:
  - Retrieval: P@k, R@k, MRR, NDCG@k (k ∈ {1,3,5})
  - Answer: semantic similarity, LLM-judge correctness (1–5)
  - Trust: hallucination rate (claim-level), faithfulness, refusal correctness
- [ ] Full 3×3 matrix runs with one command in < 30 min
- [ ] Comparison report: Markdown table + heatmap + hallucination bar +
  cost-vs-quality scatter
- [ ] 95% CI reported on aggregate metrics; no claims below CI width

### Quality
- [ ] Test coverage ≥ 80% overall, ≥ 95% on `evaluation/metrics/`
- [ ] All phase checkpoints pass in order
- [ ] Every LLM call cost-tracked and cached
- [ ] Every run reproducible via `config_hash` + `git_sha`
- [ ] `docker compose up` works from a clean clone
- [ ] README with results table and architecture diagram

---

## The 30-second pitch (for interviews)

> "I built a RAG system over company documents — but the interesting part is
> the evaluation layer. I compare 3 chunking strategies against 3 embedding
> models on a 75-question golden dataset. I defined hallucination and
> faithfulness as separate, testable metrics — hallucination is 'unsupported
> claims in the answer,' faithfulness is 'fraction of the answer grounded in
> retrieved context.' I include 15 unanswerable questions to measure refusal
> correctness. The whole matrix runs in one command, produces a heatmap and
> cost-vs-quality scatter, and every run is reproducible from a config hash.
> This is what separates a RAG demo from a RAG system."

---

## Quick start (once built)

```bash
git clone <repo> && cd doc-intelligence-platform
cp .env.example .env            # add API keys if using hosted models
make install                    # pip install -e ".[dev]"
make corpus                     # generate synthetic sample corpus
make ingest chunk embed index   # build the index
make eval-matrix                # run full comparison
make serve                      # FastAPI :8000 + Streamlit :8501