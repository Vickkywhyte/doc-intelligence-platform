# Document Intelligence Platform

> A RAG system that ingests company documents, answers questions with
> citations, and — the actual point — **measures its own quality** with a
> first-class evaluation pipeline comparing chunking strategies and embedding
> models.

[![Status](https://img.shields.io/badge/status-planning-blue)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## TL;DR

Everyone builds RAG. Almost nobody evaluates it. This project is a benchmark
harness with a chat UI on top.

- **Ingests** PDFs, DOCX, TXT, MD with page-aware offsets
- **Answers** questions with inline `[doc:page:chunk]` citations
- **Refuses** when the answer isn't in the corpus (a feature, not a bug)
- **Evaluates** every answer on a 75-question golden dataset
- **Compares** 3 chunking strategies × 3 embedding models = 9 configs
- **Reports** accuracy, hallucination rate, retrieval precision, faithfulness,
  refusal correctness — each with 95% CI
- **Reproduces** every run from a config hash + git SHA

---

## Why this exists

Three questions determine whether a RAG system is deployable:

1. **Is the answer correct?**
2. **Did the model fabricate something?**
3. **Did it retrieve the right passage?**

Most RAG demos can't answer any of them. This one can, with numbers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            INTERFACES                                │
│   Streamlit UI   │   FastAPI   │   CLI scripts   │   Notebooks       │
└─────────┬────────┴──────┬──────┴────────┬────────┴─────────┬─────────┘
          ▼               ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                           │
│   rag_pipeline.py          experiment.py          evaluator.py       │
└─────┬──────────────┬────────────────┬──────────────────┬─────────────┘
      ▼              ▼                ▼                  ▼
┌──────────┐  ┌───────────┐   ┌────────────┐   ┌────────────────┐
│Ingestion │  │ Chunking  │   │ Embeddings │   │  Evaluation    │
└────┬─────┘  └─────┬─────┘   └─────┬──────┘   └────────┬───────┘
     └──────────────┴───────────────┴───────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE / INFRA LAYER                        │
│   FileSystem (data/)   │   Chroma   │   Cost log (JSONL)             │
└─────────────────────────────────────────────────────────────────────┘
```

Full architecture in [`claude_chat/02_ARCHITECTURE.md`](claude_chat/02_ARCHITECTURE.md).

---

## Results (placeholder — filled after Phase 9)

| Chunking | Embedding | P@5 | MRR | Acc ↑ | Halluc ↓ | Faith ↑ | Refusal ↑ | Latency | $/1k |
|---|---|---|---|---|---|---|---|---|---|
| fixed | openai-small | — | — | — | — | — | — | — | — |
| fixed | bge-small | — | — | — | — | — | — | — | — |
| fixed | minilm | — | — | — | — | — | — | — | — |
| recursive | openai-small | — | — | — | — | — | — | — | — |
| recursive | bge-small | — | — | — | — | — | — | — | — |
| recursive | minilm | — | — | — | — | — | — | — | — |
| semantic | openai-small | — | — | — | — | — | — | — | — |
| semantic | bge-small | — | — | — | — | — | — | — | — |
| semantic | minilm | — | — | — | — | — | — | — | — |

Plus three charts: heatmap, hallucination bar, cost-vs-quality scatter.

---

## Quick start

```bash
git clone <repo> && cd doc-intelligence-platform
cp .env.example .env               # add OPENROUTER_API_KEY if using hosted
make install                       # pip install -e ".[dev]"
make corpus                        # generate synthetic sample PDFs
make ingest chunk embed index      # build the corpus index
make eval-matrix                   # run the full comparison (optional)
make serve                         # FastAPI :8000 + Streamlit :8501
```

Open **http://localhost:8501** → **Eval Dashboard** tab.

### Requirements

- Python 3.11
- `poppler` and `tesseract` (system deps for `unstructured`)
- Optional: OpenRouter API key ([get one](https://openrouter.ai/keys))

### Docker

```bash
docker compose up -d
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

---

## Project structure

```
doc-intelligence-platform/
├── claude_chat/          # Planning docs — source of truth
├── configs/              # YAML configs
├── data/                 # Corpus, chunks, indexes, golden set
├── results/              # Run artifacts, reports, figures
├── logs/                 # Structured logs + cost ledger
├── src/docintel/         # Library code
├── scripts/              # Numbered runnable entry points
├── api/                  # FastAPI
├── ui/                   # Streamlit
├── tests/                # Mirrors src/
├── notebooks/            # Exploration
├── pyproject.toml
├── Makefile
└── docker-compose.yml
```

---

## The 30-second pitch

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

## Documentation

All planning artifacts live in `claude_chat/`. Reading order:

**Foundations:**
- `PROJECT_INSTRUCTIONS.md` — working principles
- `CLAUDE.md` — operating rules for AI-assisted work
- `01_PRD.md` — product requirements
- `02_ARCHITECTURE.md` — system architecture
- `03_TECH_DECISIONS.md` — 14 ADRs
- `04_ROADMAP.md` — 11 phases

**Execution:**
- `05_DATA_MODEL.md` — Pydantic models + ABCs
- `06_EVAL_SPEC.md` — ⭐ the differentiator
- `07_API_SPEC.md` — REST contract
- `08_UI_SPEC.md` — Streamlit spec
- `09_TESTING_STRATEGY.md` — test pyramid
- `10_OBSERVABILITY.md` — logging + cost tracking

**Reference:**
- `11_SECURITY_PRIVACY.md` — data handling
- `12_DEPLOYMENT.md` — local + Docker + cloud path
- `13_RISKS_AND_MITIGATIONS.md` — 15 risks
- `14_GLOSSARY.md` — terminology
- `CHANGELOG.md` — decision log

---

## Status

🟡 **Planning complete.** Implementation starts at Phase 1 (Scaffolding & Config).

See [`claude_chat/04_ROADMAP.md`](claude_chat/04_ROADMAP.md) for the current
phase and [`claude_chat/00_START_HERE.md`](claude_chat/00_START_HERE.md) for
status.

---

## License

MIT (adjust as needed).