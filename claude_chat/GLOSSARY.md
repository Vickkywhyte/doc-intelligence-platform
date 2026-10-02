# Glossary

**Related docs:** `06_EVAL_SPEC.md`, `05_DATA_MODEL.md`

---

## A

**Answer accuracy** — In this project, the mean judge score (1–5) divided by
5, normalized to [0, 1]. Measured on factual and multi_hop golden items.
See `06_EVAL_SPEC.md §3.2`.

**Atomic claim** — A single verifiable factual statement extracted from an
answer during hallucination analysis. See `06_EVAL_SPEC.md §4.2`.

---

## C

**Chunk** — A contiguous span of a document used as the retrieval unit. Has
a stable `chunk_id`, char offsets into the source, and a page number.
See `05_DATA_MODEL.md §2.2`.

**Chunk ID** — `{doc_id}:{strategy}:{index:05d}`. Sortable, traceable back
to the source doc and chunking strategy.

**Chunking strategy** — The algorithm that produces chunks. We compare
three: `fixed`, `recursive`, `semantic`. See ADR-004.

**Citation** — A reference in an answer to the source chunk(s), rendered as
`[doc_id:page:chunk_id]`.

**Config hash** — SHA-256 of the canonicalized (sorted keys, no whitespace)
YAML config. Part of every `run_id`. Guarantees reproducibility.

**Cost per 1k queries** — Extrapolated from the run's total cost: `total_cost_usd / n_questions * 1000`.

**Cross-strategy gold chunk IDs** — The `gold_chunk_ids` field on a
`GoldenItem` is a dict keyed by chunking strategy, because chunk IDs differ
across strategies. See `06_EVAL_SPEC.md §1.3`.

---

## D

**Doc ID** — `sha256(file_bytes)[:16]`. Stable across machines; used to
reference a document without exposing its filename.

**Document Intelligence Platform** — This project. A RAG system with a
first-class evaluation pipeline.

---

## E

**Embedder** — A model that maps text to a dense vector. We compare three:
OpenAI `text-embedding-3-small` (hosted), BGE-small-en-v1.5 (local),
all-MiniLM-L6-v2 (local). See ADR-003.

**EvalResult** — The full artifact of one matrix run. Serialized to
`results/runs/{run_id}.json`. See `05_DATA_MODEL.md §2.5`.

**Evaluation** — In this project, the automated pipeline that measures
answer quality, hallucination, retrieval precision, and refusal correctness
against a golden dataset.

---

## F

**Faithfulness** — The fraction of statements in an answer that are
supported by retrieved context. Looser than hallucination: allows paraphrase.
See `06_EVAL_SPEC.md §5` and ADR-012.

**Fixed chunking** — Naive token-window chunking with configurable size and
overlap. The dumb baseline.

---

## G

**Golden dataset** — Human-curated Q&A pairs with ground-truth answers and
source chunk IDs. 75 items: 45 factual, 15 multi_hop, 15 unanswerable. See
`06_EVAL_SPEC.md §1` and ADR-010.

**Grounding** — The property that an answer is supported by retrieved
context.

---

## H

**Hallucination** — In this project, a claim in the answer that is not
entailed by any retrieved chunk. Operationalized via claim decomposition +
per-claim entailment check. See `06_EVAL_SPEC.md §4` and ADR-011.

---

## J

**Judge** — An LLM used to score answers or check entailment. Default:
`openai/gpt-4o-mini` at `temperature=0`, prompts pinned to versioned files.
See ADR-006.

**Judge prompt version** — A version tag on judge prompts (e.g., `v1`).
Changing a prompt creates a new version; runs record which version was used.

---

## L

**LLM-as-judge** — Using an LLM to score another LLM's output. See ADR-006.

---

## M

**Matrix run** — A single `make eval-matrix` invocation that evaluates all
3 chunking × 3 embedding = 9 configs against the golden dataset.

**MMR (Maximal Marginal Relevance)** — A retrieval diversification method
that penalizes chunks similar to already-selected ones.

**MRR (Mean Reciprocal Rank)** — The mean of 1/rank of the first relevant
chunk across questions. See `06_EVAL_SPEC.md §2.3`.

**Multi-hop question** — A golden item whose answer requires combining
information from multiple chunks or documents.

---

## N

**NDCG@k** — Normalized Discounted Cumulative Gain at k. Rewards both
relevance and good ranking. See `06_EVAL_SPEC.md §2.4`.

---

## O

**OpenRouter** — A unified API gateway to multiple LLM providers (OpenAI,
Anthropic, etc.). Used as the transport layer for hosted models. Model names
use the format `provider/model`.

---

## P

**P@k (Precision at k)** — The fraction of top-k retrieved chunks that are
relevant. See `06_EVAL_SPEC.md §2.1`.

---

## Q

**QuestionResult** — The per-question slice of an `EvalResult`. Includes the
answer, retrieved chunks, all metrics, hallucinated claims, judge verdict,
cost, and latency. See `05_DATA_MODEL.md §2.5`.

---

## R

**RAG (Retrieval-Augmented Generation)** — Combining retrieval with an LLM
to produce grounded answers.

**Recursive chunking** — Chunking that tries a hierarchy of separators
(paragraph → sentence → word) until chunks fit the target size. The industry
default.

**Refusal** — An explicit "I don't know" response when the corpus lacks the
answer. A feature, not a bug. See ADR-013.

**Refusal correctness** — The fraction of unanswerable golden items where
the system correctly refused. See `06_EVAL_SPEC.md §3.3`.

**Reranker** — A second-stage model (usually a cross-encoder) that reorders
the top-N retrieved chunks. Feature-flagged; deferred to v1.1.

**Retrieval** — Selecting the top-k chunks most relevant to a query.

**Run ID** — `{YYYYMMDDTHHMMSS}_{config_hash[:8]}`. Unique identifier for
one matrix run.

---

## S

**Semantic chunking** — Chunking that splits on embedding-similarity
valleys between sentences. Premium; tests the hypothesis that chunking
matters.

**Semantic similarity** — `cosine(embed(gold_answer), embed(generated_answer))`.
Reported as `sem_sim`. Never used alone. See `06_EVAL_SPEC.md §3.1`.

**structlog** — The structured logging library used across the project. JSON
output, queryable on disk.

---

## T

**Top-k retrieval** — Standard retrieval: return the k highest-scoring
chunks by vector similarity.

---

## U

**Unanswerable question** — A golden item whose answer is not in the corpus.
Tests refusal correctness. 15/75 = 20% of the golden set.

---

## V

**Vector store** — Persistence layer for embeddings + metadata. Chroma in
v1. See ADR-002.

---

## W

**Warning** — A non-fatal flag attached to a `RAGResponse` or `EvalResult`.
Examples: `low_retrieval_score`, `zero_claims`, `nondeterministic_provider`.

---

## Numbers

**3×3 matrix** — 3 chunking strategies × 3 embedding models = 9 configs
evaluated in a single `make eval-matrix` run.

**75 questions** — Size of the golden dataset: 45 factual + 15 multi_hop +
15 unanswerable.

**95% CI** — Bootstrap confidence interval on each aggregate metric. Report
includes it; differences smaller than the CI half-width are not claimed.