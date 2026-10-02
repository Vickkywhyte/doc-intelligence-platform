# Evaluation Specification

**Related docs:** `01_PRD.md`, `05_DATA_MODEL.md`, `03_TECH_DECISIONS.md` (ADR-011, ADR-012, ADR-013)

> ⭐ This is the differentiator. Read carefully. Every metric has a precise
> definition, a formula, a unit-test reference, and a documented failure mode.

---

## 1. Golden dataset

### 1.1 Schema (per line, JSONL)

```json
{
  "id": "q001",
  "question": "What is the notice period for termination without cause?",
  "gold_answer": "30 days written notice.",
  "gold_chunk_ids": {
    "fixed":     ["a1b2...:fixed:00007"],
    "recursive": ["a1b2...:recursive:00012"],
    "semantic":  ["a1b2...:semantic:00005"]
  },
  "type": "factual",
  "notes": "Section 7.2 of MSA"
}
```

### 1.2 Composition rules

| Type | Count | Purpose |
|---|---|---|
| `factual` | 45 | Baseline accuracy |
| `multi_hop` | 15 | Tests retrieval across chunks/docs |
| `unanswerable` | 15 | Tests refusal correctness |
| **Total** | **75** | |

- Every `gold_chunk_id` **must exist** in `data/chunks/{strategy}/` for **all
  three strategies** (see §1.3).
- Every `gold_answer` ≤ 2 sentences. Long answers amplify judge noise.
- Unanswerable entries have `gold_chunk_ids: {}` and a `notes` field
  explaining what's missing from the corpus.

### 1.3 Cross-strategy ground truth (why the dict?)

Chunk IDs are **strategy-specific** — a fact on page 7 will land in
`a1b2...:fixed:00007`, `a1b2...:recursive:00012`, and `a1b2...:semantic:00005`
depending on the chunker. Retrieval metrics must compare against the slice
matching the current config's chunker:

```python
gold_for_this_config = item.gold_chunk_ids[config.chunking.name]
```

If a golden entry omits a strategy, `make golden-validate` fails.

### 1.4 Curation process

1. `scripts/05_seed_golden.py` proposes questions from parsed docs by text
   overlap between candidate answers and chunk text.
2. Human edits each entry: verifies question, writes gold answer, confirms
   chunk mappings for all three strategies.
3. Committed via PR. No script modifies `golden.jsonl` after initial seed.
4. `make golden-validate` enforces schema + coverage on every change.

---

## 2. Retrieval metrics

Let:
- `R` = ranked list of retrieved chunk IDs (top-k, position 1 = best)
- `G` = set of gold chunk IDs for the current strategy
- `k` ∈ {1, 3, 5}

### 2.1 Precision@k

```
P@k = |{top-k of R} ∩ G| / k
```

- Range: [0, 1]
- Reported per question, then averaged as `p_at_{k}`.
- If `G` is empty (unanswerable): **N/A**, excluded from aggregates.

### 2.2 Recall@k

```
R@k = |{top-k of R} ∩ G| / |G|
```

- Range: [0, 1]
- Averaged as `recall_at_{k}`.
- If `G` is empty: N/A.

### 2.3 MRR (Mean Reciprocal Rank)

```
RR  = 1 / rank_of_first_relevant   (0 if none in top-k)
MRR = mean(RR) across questions
```

- Range: [0, 1]
- Only the **first** relevant chunk matters; subsequent ranks ignored.
- Averaged as `mrr`.

### 2.4 NDCG@k

Standard formulation with **binary relevance** (chunk is gold or not):

```
DCG@k  = Σ_{i=1..k}  rel_i / log2(i + 1)
IDCG@k = Σ_{i=1..min(|G|,k)} 1 / log2(i + 1)
NDCG@k = DCG@k / IDCG@k   (0 if IDCG@k == 0)
```

- Range: [0, 1]
- Rewards both relevance *and* good ranking.
- Averaged as `ndcg_at_{k}`.

### 2.5 Test fixtures (must pass)

| Fixture | Setup | Expected |
|---|---|---|
| `perfect_retrieval` | G={c1}, R=[c1, c2, c3, c4, c5] | P@5=0.2, MRR=1.0, NDCG@5=0.469 |
| `gold_at_rank_4` | G={c4}, R=[c1,c2,c3,c4,c5] | MRR=0.25 |
| `multi_gold` | G={c1,c3}, R=[c1,c2,c3,c4,c5] | P@5=0.4, R@5=1.0, MRR=1.0 |
| `no_gold_retrieved` | G={c9}, R=[c1..c5] | P@5=0.0, MRR=0.0 |
| `unanswerable` | G={}, R=[c1..c5] | All N/A → excluded |

---

## 3. Answer metrics

### 3.1 Semantic similarity

```
sem_sim = cosine(embed(gold_answer), embed(generated_answer))
```

- Uses the **matrix's own embedder** (so it's comparable across configs).
- Range: [-1, 1] but practically [0, 1] for natural language.
- Averaged as `sem_sim`.

**Failure mode:** Reworded-but-correct answers can score < 0.8. **Never used
alone.** Always reported alongside `answer_accuracy`.

### 3.2 Correctness (LLM judge, 1–5)

**Prompt** (`evaluation/judge_prompts/correctness_v1.txt`):

```
You are grading whether an ANSWER correctly responds to a QUESTION given a
reference GOLD_ANSWER. Consider meaning, not wording. Ignore style, length,
and citation format.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {answer}

Return ONLY JSON: {"score": <1-5>, "rationale": "<one sentence>"}

Score guide:
1 = wrong or contradicts gold
2 = mostly wrong but a fragment is right
3 = partially correct, missing key detail
4 = correct but with a minor extra or omission
5 = fully correct, no missing essential info
```

**Aggregation:**
```
answer_accuracy = mean(score) / 5     → normalized to [0, 1]
```

- Only scored on `factual` and `multi_hop` items.
- Unanswerables are excluded (they're scored by refusal correctness).

### 3.3 Refusal correctness (unanswerable only)

```
refusal_correctness = (# unanswerable items where response.refused == True)
                    / (# unanswerable items)
```

- Range: [0, 1]
- Target: ≥ 0.90

**False refusal cost:** A system that refuses everything would score
`refusal_correctness = 1.0` but tank `answer_accuracy`. Both are reported;
neither is optimized in isolation.

### 3.4 Refusal detection

A response is marked `refused=True` if **either**:
1. The answer contains an explicit refusal phrase (case-insensitive, from a
   curated list: `"i don't know"`, `"i cannot answer"`, `"not in the provided
   documents"`, `"no relevant information"`, etc.), **or**
2. The response has zero citations AND zero retrieved chunks above the
   configured score floor.

The curated phrase list lives in `generation/prompt_templates.py` and is
unit-tested on positive and negative cases.

---

## 4. Hallucination (ADR-011)

### 4.1 Definition

A claim is **hallucinated** iff it appears in the answer but is **not
entailed** by any retrieved chunk.

### 4.2 Algorithm

**Step 1 — Claim decomposition** (LLM, temp=0):
```
Prompt: claim_decomposition_v1.txt

Decompose the ANSWER into atomic factual claims. An atomic claim is a single
verifiable statement. Ignore:
- restatements of the question
- hedges, transitions, meta-commentary
- citations in square brackets

Return ONLY JSON: {"claims": ["<claim 1>", "<claim 2>", ...]}

ANSWER: {answer}
```

**Step 2 — Entailment check per claim** (LLM, temp=0):
```
Prompt: entailment_v1.txt

Is the CLAIM entailed by ANY of the provided CHUNKS?
Answer: yes | partial | no.
- yes     = a chunk fully supports the claim
- partial = a chunk supports part of the claim
- no      = no chunk supports the claim

Return ONLY JSON: {"verdict": "yes|partial|no", "chunk_id": "<id or null>"}

CLAIM: {claim}

CHUNKS:
[chunk_id_1] {text_1}
[chunk_id_2] {text_2}
...
```

**Step 3 — Rate** (per question):
```
hallucination_rate_q = (n_no + partial_claim_weight * n_partial) / n_claims
```

- `partial_claim_weight` defaults to 0.5 (configurable in `evaluation.*`).
- Aggregate as mean across all scored questions: `hallucination_rate`.

### 4.3 Exclusions (documented behavior)

- Claims that restate the question → not counted.
- Claims in a refusal → answer has 0 claims → rate = 0 (flagged in report).
- If an answer has zero atomic claims and is not a refusal → rate = 0,
  but the question is flagged in `per_question[].judge_verdict` with
  `{"warning": "zero_claims"}`.

### 4.4 Why not single-shot?

A single "is this hallucinated?" prompt is opaque and inconsistent. Claim
decomposition makes each verdict:
- **Inspectable** — every claim + verdict + supporting chunk is logged.
- **Cacheable** — by claim hash.
- **Testable** — fixtures can inject known claims and expected verdicts.

### 4.5 Test fixtures

| Fixture | Answer | Chunks | Expected rate |
|---|---|---|---|
| `clean` | "Notice is 30 days." | "Either party may terminate upon 30 days' notice." | 0.0 |
| `fabricated_number` | "Notice is 60 days." | "Either party may terminate upon 30 days' notice." | 1.0 |
| `partial` | "Notice is 30 days, effective immediately." | "...upon 30 days' notice." | 0.5 |
| `refusal` | "I don't know based on the provided documents." | (any) | 0.0 |
| `empty` | "" | (any) | 0.0, flagged |

---

## 5. Faithfulness (ADR-012)

### 5.1 Definition

Faithfulness measures whether the retrieved context **justifies** the answer.
Distinct from hallucination:
- **Hallucination** is strict: claims not entailed.
- **Faithfulness** is looser: statements *supported* by context, allowing
  paraphrase and summarization.

### 5.2 Operationalization

Reuses claim decomposition from §4.2 Step 1 (cached). For each claim:
```
faithfulness_q = (# claims supported by retrieved context)
               / (# claims)
```
"Supported" = entailment verdict in `{yes, partial}` (both count as
supported for faithfulness; contrast with hallucination where partial is
half-penalized).

### 5.3 Interpretation

- High hallucination, high faithfulness: model paraphrases then adds extras.
- Low hallucination, low faithfulness: answer is correct but not grounded in
  retrieved content (retrieval problem).
- Both should be high for a well-behaved system.

### 5.4 Test fixtures

| Fixture | Answer | Chunks | Faithfulness |
|---|---|---|---|
| `paraphrase` | "Either side can end the deal with a month's notice." | "Either party may terminate upon 30 days' notice." | 1.0 |
| `unsupported_extra` | "Notice is 30 days and payment is due on signing." | "...upon 30 days' notice." | 0.5 |

---

## 6. Aggregate metrics (the report table)

| Metric | Definition | Direction | Applies to |
|---|---|---|---|
| `p_at_5` | mean P@5 | ↑ | factual, multi_hop |
| `recall_at_5` | mean R@5 | ↑ | factual, multi_hop |
| `mrr` | mean RR | ↑ | factual, multi_hop |
| `ndcg_at_5` | mean NDCG@5 | ↑ | factual, multi_hop |
| `answer_accuracy` | mean judge score / 5 | ↑ | factual, multi_hop |
| `sem_sim` | mean cosine sim | ↑ | factual, multi_hop |
| `hallucination_rate` | mean claim-level hallucination | ↓ | factual, multi_hop |
| `faithfulness` | mean statement-level support | ↑ | factual, multi_hop |
| `refusal_correctness` | unanswerable refusal rate | ↑ | unanswerable |
| `latency_ms_p50` | median end-to-end latency | ↓ | all |
| `cost_per_1k_q_usd` | extrapolated cost | ↓ | all |

### 6.1 The headline table

`results/reports/comparison.md` renders:

| Chunking | Embedding | P@5 | MRR | Acc | Halluc ↓ | Faith | Refusal | Latency | $/1k |
|---|---|---|---|---|---|---|---|---|---|
| fixed | openai-small | ... | ... | ... | ... | ... | ... | ... | ... |
| fixed | bge-small | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 7. Statistical hygiene

### 7.1 Judge versioning
Every run records:
- `judge_model` (e.g., `openai/gpt-4o-mini`)
- `judge_prompt_version` (e.g., `v1`)

Changing a judge prompt = new version + new ADR-worthy decision. Do not edit
`*_v1.txt` in place — create `*_v2.txt`.

### 7.2 Determinism
- `temperature=0` on all judge calls.
- `seed=config.evaluation.seed` on provider calls that support it.
- Judge verdicts cached by `hash(prompt + model + prompt_version)`.
- Report any provider that does not support deterministic output
  (as a run-level `warnings` entry).

### 7.3 Self-preference mitigation
- Generator and judge are different model families when possible.
- At minimum, judge is a **stronger** model than generator (e.g., GPT-4o judge
  for a GPT-4o-mini generator).

### 7.4 Confidence intervals
For each aggregate metric, bootstrap 1000× on per-question scores:
```
ci_95 = [p2.5, p97.5] of bootstrap distribution
```
- Report `[lo, hi]` half-width alongside each aggregate.
- **Do not claim differences smaller than CI width** in the report.

### 7.5 Nondeterminism disclosure
Every report header includes:
```
Run: {run_id}
Judge: {model} (prompt {version}, temp=0)
Generator: {model} (temp={t})
Seed: {seed}
Nondeterminism sources: {list}
```

---

## 8. Test fixtures summary

`tests/test_metrics.py` must include these hand-crafted cases (from §2.5,
§4.5, §5.4) and assert expected values within `1e-6`:

**Retrieval:**
- `perfect_retrieval`
- `gold_at_rank_4`
- `multi_gold`
- `no_gold_retrieved`
- `unanswerable`

**Answer:**
- `perfect_answer` → accuracy = 1.0
- `wrong_answer` → accuracy ≤ 0.2
- `refusal_on_answerable` → accuracy ≤ 0.4
- `refusal_on_unanswerable` → refusal_correctness = 1.0

**Hallucination:**
- `clean`, `fabricated_number`, `partial`, `refusal`, `empty`

**Faithfulness:**
- `paraphrase` → 1.0
- `unsupported_extra` → 0.5

Any metric shipped without all its fixtures is a **blocker** for Phase 8 sign-off.

---

## 9. What this spec deliberately does NOT do

- **Not** measure factuality against a knowledge base beyond the corpus.
  We evaluate *groundedness against retrieved context*, not world truth.
- **Not** score tone, style, or fluency. Those aren't our failure modes.
- **Not** merge hallucination and faithfulness. See ADR-012.
- **Not** auto-generate gold answers at run time. See ADR-010.
- **Not** compare across runs with different judge versions without flagging
  it in the report header.

---

## 10. Sign-off checklist for Phase 8

- [ ] All metrics implemented behind `Metric` ABC
- [ ] All fixtures from §8 pass at 1e-6 tolerance
- [ ] Judge prompts pinned to `_v1.txt` files
- [ ] Judge cache working (2nd run = 0 API calls for judge)
- [ ] `make eval-single` produces a complete `EvalResult`
- [ ] Bootstrap CI computed and stored in `aggregate_ci`
- [ ] Report header includes run metadata + nondeterminism disclosure
- [ ] Any deviation from this spec captured as a new ADR