# Golden Dataset — Curation Rules and Coverage Report

## Purpose

`golden.jsonl` is the human-curated evaluation dataset for the Document
Intelligence Platform.  Every automated evaluation run compares system
outputs against these entries.  The quality of the dataset directly
determines the validity of all downstream metrics.

**Do not modify `golden.jsonl` programmatically** — see `CLAUDE.md §5`.

---

## File locations

| File | Purpose |
|---|---|
| `data/eval/golden.jsonl` | Production golden dataset (75 entries) |
| `data/eval/golden.proposed.jsonl` | Seeder output — PENDING HUMAN REVIEW |

---

## Dataset composition

| Type | Count | Description |
|---|---|---|
| `factual` | 45 | Single-doc facts; one chunk contains the full answer |
| `multi_hop` | 15 | Require combining info across ≥ 2 chunks or documents |
| `unanswerable` | 15 | Not present in corpus; correct answer is a refusal |
| **Total** | **75** | |

---

## Schema (GoldenItem)

```jsonc
{
  "id":             "g001",               // unique, e.g. g001–g075
  "question":       "...",                // natural-language question
  "gold_answer":    "...",                // canonical answer (≤ 2 sentences)
  "gold_chunk_ids": {                     // empty dict for unanswerable
    "fixed":     ["<doc_id>:fixed:<idx>", ...],
    "recursive": ["<doc_id>:recursive:<idx>", ...],
    "semantic":  ["<doc_id>:semantic:<idx>", ...]
  },
  "type":  "factual" | "multi_hop" | "unanswerable",
  "notes": "optional curation note"
}
```

`gold_chunk_ids` must cover **all three strategies** for `factual` and
`multi_hop` entries.  The cross-strategy mapping allows the evaluator to
compute retrieval metrics (P@k, R@k, MRR, NDCG@k) independently per
strategy — see `06_EVAL_SPEC.md §1.3`.

---

## Corpus coverage

The 75 entries reference content from all 12 corpus documents:

| Document | doc_id prefix | Doc type | Entries |
|---|---|---|---|
| TechCorp DPA — Acme Corporation | `2f81405b` | dpa | 5 factual + 1 multi_hop |
| TechCorp Remote Work Policy | `482669c1` | policy | 3 factual + 4 multi_hop |
| Contractor Handbook | `4a14ce09` | handbook | 2 factual + 4 multi_hop |
| TechCorp DPA — Globex Industries | `6b70daba` | dpa | 5 factual + 1 multi_hop |
| MSA — Globex Industries, Inc. | `801091c0` | contract | 5 factual + 8 multi_hop |
| Information Security Policy | `a2f367c9` | policy | 3 factual + 5 multi_hop |
| MSA — Initech Corporation | `ae958268` | contract | 5 factual + 3 multi_hop |
| MSA — Acme Corporation | `b6befe3b` | contract | 5 factual + 4 multi_hop |
| Human Resources Policy | `c6fd7418` | policy | 2 factual + 1 multi_hop |
| NDA — Acme Corporation | `df286937` | nda | 4 factual + 3 multi_hop |
| Employee Handbook | `fcc833fc` | handbook | 2 factual + 4 multi_hop |
| NDA — Globex Industries, Inc. | `fdd20828` | nda | 4 factual + 3 multi_hop |

---

## Curation rules

### For factual entries

1. The answer must be directly supported by a **single passage** in one
   document.
2. `gold_answer` must be ≤ 2 sentences; copy the relevant phrase verbatim
   where possible.
3. Map the answer to chunk IDs for **all 3 strategies** using char-offset
   overlap: a chunk qualifies if `chunk.char_start < answer_end AND
   chunk.char_end > answer_start`.
4. Do not paraphrase; if the document text is unambiguous, use it directly.

### For multi-hop entries

1. The answer requires combining information from ≥ 2 distinct chunks or
   documents.
2. List every chunk that a retriever **must** return for the answer to be
   complete in `gold_chunk_ids`.
3. The `notes` field must name the documents or chunks that must be combined.

### For unanswerable entries

1. Verify the question is **genuinely absent** from all 12 documents before
   adding.
2. `gold_answer` must be exactly: `"This information is not available in the
   provided documents."`
3. `gold_chunk_ids` must be `{}`.
4. The `notes` field must state why the topic is absent.

### General

- Questions must be grammatically correct and unambiguous.
- Do not add questions whose answers appear in the document title or heading
  alone — they must require reading body text.
- Review the `06_EVAL_SPEC.md` refusal definition before adding/removing
  unanswerable entries; the threshold affects refusal-correctness metrics.

---

## How to add or remove entries

1. Edit `golden.jsonl` directly (it is not auto-generated).
2. Run `make golden-validate` to verify schema and chunk ID integrity.
3. Commit with a message: `[phase-7] eval(golden): <description>`.
4. If adding multi-hop entries, fill `gold_chunk_ids` manually by running:
   ```bash
   python scripts/05_seed_golden.py  # proposes candidates
   ```
   then promoting the relevant proposed entries.

---

## Regenerating from scratch

```bash
python scripts/05_seed_golden.py      # writes golden.proposed.jsonl
# Human review: prune to 75, fill multi_hop chunk IDs
# Copy to golden.jsonl
make golden-validate                  # verify
```

The seeder script is **idempotent** — re-running it overwrites
`golden.proposed.jsonl` but never touches `golden.jsonl`.
