"""Phase 7 — Propose golden-dataset candidates from the existing corpus.

Output: data/eval/golden.proposed.jsonl  (PENDING HUMAN REVIEW — not golden.jsonl)

Workflow:
  1. Run this script to generate candidates.
  2. Human edits each entry: verifies answer, confirms chunk mappings.
  3. Manually copy approved entries to data/eval/golden.jsonl.
  4. Run `make golden-validate` to enforce schema and coverage.

NEVER run this script against golden.jsonl (CLAUDE.md §5).

Run: python scripts/05_seed_golden.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Helpers ───────────────────────────────────────────────────────────────────

STRATEGIES = ["fixed", "recursive", "semantic"]
CHUNKS_DIR = Path("data/chunks")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = Path("data/eval/golden.proposed.jsonl")
PENDING_MARKER = "PENDING REVIEW — verify answer and chunk IDs before promoting"


def _load_chunks() -> dict[str, list[dict]]:
    """Load all chunks keyed by strategy."""
    result: dict[str, list[dict]] = {}
    for strategy in STRATEGIES:
        chunks: list[dict] = []
        for jsonl in sorted((CHUNKS_DIR / strategy).glob("*.jsonl")):
            for line in jsonl.read_text().splitlines():
                if line.strip():
                    chunks.append(json.loads(line))
        result[strategy] = chunks
    return result


def _chunks_for_span(
    chunks_by_strategy: dict[str, list[dict]],
    doc_id: str,
    char_start: int,
    char_end: int,
) -> dict[str, list[str]]:
    """Return {strategy: [chunk_id, ...]} for chunks that overlap [char_start, char_end)."""
    gold_ids: dict[str, list[str]] = {}
    for strategy, all_chunks in chunks_by_strategy.items():
        matching = [
            c["chunk_id"]
            for c in all_chunks
            if c["doc_id"] == doc_id
            and c["char_start"] < char_end  # chunk starts before span ends
            and c["char_end"] > char_start  # chunk ends after span starts
        ]
        if matching:
            gold_ids[strategy] = sorted(set(matching))
    return gold_ids


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _find_span(full_text: str, answer_text: str) -> tuple[int, int] | None:
    idx = full_text.find(answer_text)
    if idx == -1:
        return None
    return idx, idx + len(answer_text)


# ── Per-doc-type fact patterns ────────────────────────────────────────────────


def _extract_contract(doc: dict, full_text: str) -> list[dict]:
    facts = []
    doc_id = doc["doc_id"]
    customer = doc.get("metadata", {}).get("counterparty") or _guess_customer(full_text)

    _PATTERNS = [
        # (answer_regex, question_template)
        (
            r"(?:Customer shall pay|invoiced|fees.*?within)\s+[^\n]{10,80}(?:days?|net-\d+)[^\n]{0,40}",
            "What are the payment terms under the Master Services Agreement?",
        ),
        (
            r"Either party may terminate[^\n]{20,120}",
            "Under what conditions can either party terminate the agreement?",
        ),
        (
            r"(?:governed by|construed in accordance with)[^\n]{20,80}",
            "What law governs the Master Services Agreement?",
        ),
        (
            r"(?:limitation of liability|liability.*?shall not exceed)[^\n]{20,120}",
            "What is the limitation of liability clause in the agreement?",
        ),
        (
            r"(?:warranty|warrants)[^\n]{20,100}",
            "What warranty does the Service Provider provide?",
        ),
        (
            r"(?:Intellectual Property|IP|assigns to)[^\n]{20,100}",
            "Who owns the intellectual property created under the agreement?",
        ),
        (
            r"(?:indemnif)[^\n]{20,100}",
            "What are the indemnification obligations of the parties?",
        ),
        (
            r"(?:Confidential Information)[^\n]{20,100}",
            "How is Confidential Information defined in the MSA?",
        ),
        (
            r"(?:force majeure|Force Majeure)[^\n]{20,100}",
            "What events are covered by the force majeure clause?",
        ),
        (
            r"(?:taxes|tax)[^\n]{20,80}",
            "Who is responsible for taxes under the agreement?",
        ),
    ]

    for pattern, question in _PATTERNS:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if not m:
            continue
        answer_text = m.group(0).strip().rstrip(".,;")
        span = _find_span(full_text, answer_text)
        if not span:
            continue
        facts.append(
            {
                "answer_text": answer_text,
                "char_start": span[0],
                "char_end": span[1],
                "question": question,
                "doc_id": doc_id,
                "doc_type": "contract",
            }
        )
    return facts


def _extract_nda(doc: dict, full_text: str) -> list[dict]:
    doc_id = doc["doc_id"]
    _PATTERNS = [
        (
            r"(?:Confidential Information.*?means|\"Confidential Information\".*?includes)[^\n]{20,150}",
            "What is defined as Confidential Information in the NDA?",
        ),
        (
            r"obligations.*?do not apply[^\n]{20,150}",
            "What information is excluded from the confidentiality obligations?",
        ),
        (
            r"(?:term|period|duration).*?(?:two|three|five|\d) years?[^\n]{0,60}",
            "How long do the confidentiality obligations last under the NDA?",
        ),
        (
            r"(?:disclose|disclosure)[^\n]{20,100}",
            "Under what circumstances may Confidential Information be disclosed?",
        ),
        (
            r"(?:governing law|governed by)[^\n]{20,80}",
            "What governing law applies to the non-disclosure agreement?",
        ),
        (
            r"(?:injunctive relief|equitable)[^\n]{20,100}",
            "What remedies are available for breach of the NDA?",
        ),
    ]
    facts = []
    for pattern, question in _PATTERNS:
        m = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        answer_text = m.group(0).strip()[:200].rstrip(".,;")
        span = _find_span(full_text, answer_text)
        if not span:
            continue
        facts.append(
            {
                "answer_text": answer_text,
                "char_start": span[0],
                "char_end": span[1],
                "question": question,
                "doc_id": doc_id,
                "doc_type": "nda",
            }
        )
    return facts


def _extract_dpa(doc: dict, full_text: str) -> list[dict]:
    doc_id = doc["doc_id"]
    _PATTERNS = [
        (
            r'"Controller"[^\n]{20,120}',
            "How is 'Controller' defined in the Data Processing Agreement?",
        ),
        (
            r'"Processor"[^\n]{20,120}',
            "How is 'Processor' defined in the Data Processing Agreement?",
        ),
        (
            r'"Personal Data"[^\n]{20,120}',
            "How is 'Personal Data' defined in the Data Processing Agreement?",
        ),
        (
            r"(?:data breach|security incident)[^\n]{20,100}",
            "What must the Processor do in the event of a data breach?",
        ),
        (
            r"(?:sub-processor|subprocessor)[^\n]{20,100}",
            "What restrictions apply to the use of sub-processors?",
        ),
        (
            r"(?:data retention|retain.*?data|delete.*?data)[^\n]{20,100}",
            "What are the data retention and deletion obligations under the DPA?",
        ),
        (
            r"(?:GDPR|data protection law)[^\n]{20,100}",
            "What data protection laws govern the data processing?",
        ),
        (
            r"(?:security measures?|technical.*?organizational)[^\n]{20,100}",
            "What security measures must the Processor implement?",
        ),
    ]
    facts = []
    for pattern, question in _PATTERNS:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if not m:
            continue
        answer_text = m.group(0).strip()[:200].rstrip(".,;")
        span = _find_span(full_text, answer_text)
        if not span:
            continue
        facts.append(
            {
                "answer_text": answer_text,
                "char_start": span[0],
                "char_end": span[1],
                "question": question,
                "doc_id": doc_id,
                "doc_type": "dpa",
            }
        )
    return facts


def _extract_policy(doc: dict, full_text: str) -> list[dict]:
    doc_id = doc["doc_id"]
    _PATTERNS = [
        (
            r"(?:effective|version)[^\n]{10,80}\d{4}",
            "When does this policy take effect?",
        ),
        (
            r"(?:eligible|eligib)[^\n]{20,120}",
            "Who is eligible under this policy?",
        ),
        (
            r"(?:must|shall|required)[^\n]{20,100}",
            "What are the key requirements specified in this policy?",
        ),
        (
            r"(?:password|access control|encryption)[^\n]{20,100}",
            "What security requirements are specified for access control?",
        ),
        (
            r"(?:violation|non-compliance|disciplinary)[^\n]{20,100}",
            "What are the consequences for policy violations?",
        ),
        (
            r"(?:approved|authorized)[^\n]{20,100}",
            "What must employees obtain approval for under this policy?",
        ),
    ]
    facts = []
    for pattern, question in _PATTERNS:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if not m:
            continue
        answer_text = m.group(0).strip()[:200].rstrip(".,;")
        span = _find_span(full_text, answer_text)
        if not span:
            continue
        facts.append(
            {
                "answer_text": answer_text,
                "char_start": span[0],
                "char_end": span[1],
                "question": question,
                "doc_id": doc_id,
                "doc_type": "policy",
            }
        )
    return facts


def _extract_handbook(doc: dict, full_text: str) -> list[dict]:
    doc_id = doc["doc_id"]
    _PATTERNS = [
        (
            r"(?:code of conduct|conduct)[^\n]{20,100}",
            "What conduct expectations are stated in the handbook?",
        ),
        (
            r"(?:intellectual property|IP|work product)[^\n]{20,100}",
            "What is the policy on intellectual property created during engagement?",
        ),
        (
            r"(?:termination|terminate)[^\n]{20,100}",
            "What are the conditions for termination described in the handbook?",
        ),
        (
            r"(?:conflict of interest)[^\n]{20,100}",
            "What conflicts of interest must be disclosed or avoided?",
        ),
        (
            r"(?:compliance|law|regulation)[^\n]{20,100}",
            "What compliance obligations are described in the handbook?",
        ),
    ]
    facts = []
    for pattern, question in _PATTERNS:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if not m:
            continue
        answer_text = m.group(0).strip()[:200].rstrip(".,;")
        span = _find_span(full_text, answer_text)
        if not span:
            continue
        facts.append(
            {
                "answer_text": answer_text,
                "char_start": span[0],
                "char_end": span[1],
                "question": question,
                "doc_id": doc_id,
                "doc_type": "handbook",
            }
        )
    return facts


def _guess_customer(text: str) -> str:
    m = re.search(r"and ([\w\s,\.]+(?:Inc\.|LLC|Corp|Corporation|Industries))", text)
    return m.group(1).strip() if m else "the Customer"


# ── Unanswerable candidates ───────────────────────────────────────────────────

_UNANSWERABLE = [
    "What is TechCorp's cryptocurrency reimbursement policy?",
    "What is the annual bonus structure for employees?",
    "What is the duration of paid parental leave?",
    "What are the overtime pay rates for hourly employees?",
    "What is the per-diem travel reimbursement limit?",
    "Which cloud storage services are approved for storing company data?",
    "How many vacation days do independent contractors receive?",
    "What is the process for filing a workers' compensation claim?",
    "What is the company's policy on outside employment or moonlighting?",
    "What is the minimum age requirement for employment at TechCorp?",
    "What is the required data backup frequency for production systems?",
    "What insurance coverage must independent contractors maintain?",
    "What is the company's policy on software license audits?",
    "What severance package is provided upon termination without cause?",
    "What is the required notice period for a contractor ending their engagement?",
]

_UNANSWERABLE_NOTES = [
    "No cryptocurrency or digital asset policy exists in the corpus.",
    "No bonus structure or variable compensation details in the corpus.",
    "Parental leave is not covered in any policy or handbook.",
    "Overtime pay rates are not specified anywhere in the corpus.",
    "Travel reimbursement rates and per-diems are absent from the corpus.",
    "No approved software/cloud service list exists in the corpus.",
    "Contractor vacation/PTO entitlements are not described anywhere.",
    "Workers' compensation procedures are not covered in the corpus.",
    "No moonlighting or secondary employment policy in the corpus.",
    "No age requirements or equal opportunity specifics in the corpus.",
    "Backup frequency for systems is not covered in the corpus.",
    "Contractor insurance requirements are not specified in the corpus.",
    "Software license audit procedures are not covered in the corpus.",
    "No severance terms are specified in any document in the corpus.",
    "Contractor-side notice period for ending engagements is not covered.",
]

# ── Multi-hop candidates (require combining information across chunks/docs) ───

_MULTI_HOP = [
    {
        "question": "Which MSA customers are also subject to a separate non-disclosure agreement with TechCorp?",
        "gold_answer": "Acme Corporation and Globex Industries, Inc. have both MSAs and NDAs with TechCorp Solutions.",
        "notes": "Multi-hop: requires matching customer names across contract and NDA documents.",
    },
    {
        "question": "Do the DPAs reference the same confidentiality definition as the NDAs?",
        "gold_answer": "Both DPAs and NDAs define Confidential Information broadly but use separate document-specific definitions.",
        "notes": "Multi-hop: compare DPA Personal Data definition to NDA Confidential Information definition.",
    },
    {
        "question": "Are the governing law provisions in the MSAs consistent across all three agreements?",
        "gold_answer": "All three MSAs specify the same governing law in their respective agreements.",
        "notes": "Multi-hop: compare governing law clause across all three MSA documents.",
    },
    {
        "question": "Which documents in the corpus apply to independent contractors specifically?",
        "gold_answer": "The Contractor Handbook applies specifically to independent contractors engaged by TechCorp.",
        "notes": "Multi-hop: identify documents that explicitly scope to contractors vs. employees.",
    },
    {
        "question": "Do both NDAs include the same exclusions from the confidentiality obligations?",
        "gold_answer": "Both NDAs include the same standard exclusions, including information that becomes publicly known.",
        "notes": "Multi-hop: compare exclusions clause across the two NDA documents.",
    },
    {
        "question": "Which policies apply to both employees and contractors according to the corpus?",
        "gold_answer": "The Information Security Policy applies to all employees, contractors, and third-party vendors.",
        "notes": "Multi-hop: requires reading multiple policy scope statements.",
    },
    {
        "question": "Do the MSAs for Acme Corporation and Globex Industries have the same effective date?",
        "gold_answer": "No; the Acme MSA is effective January 1, 2024 and the Globex MSA is effective March 15, 2024.",
        "notes": "Multi-hop: compare effective dates across two MSA documents.",
    },
    {
        "question": "Are the data processing obligations for Acme Corporation and Globex Industries identical in their respective DPAs?",
        "gold_answer": "The two DPAs follow the same structure and obligations for Processor and Controller roles.",
        "notes": "Multi-hop: compare DPA obligations across the two DPA documents.",
    },
    {
        "question": "What is the relationship between TechCorp's Information Security Policy and its contractual confidentiality obligations?",
        "gold_answer": "The Information Security Policy establishes internal controls that support the confidentiality obligations in the NDAs and MSAs.",
        "notes": "Multi-hop: requires reading the security policy scope and MSA/NDA confidentiality clauses.",
    },
    {
        "question": "Do any of the MSAs reference the data processing agreement as a related document?",
        "gold_answer": "The MSAs do not explicitly reference a separate DPA, though both exist for overlapping counterparties.",
        "notes": "Multi-hop: cross-reference MSA and DPA for same customers (Acme, Globex).",
    },
    {
        "question": "Which documents in the corpus contain explicit termination provisions?",
        "gold_answer": "The MSAs and the Employee and Contractor Handbooks contain explicit termination provisions.",
        "notes": "Multi-hop: identify termination clauses across all 12 documents.",
    },
    {
        "question": "Are the effective dates of both NDAs in 2024?",
        "gold_answer": "Yes; the Acme NDA is effective February 1, 2024 and the Globex NDA is effective April 10, 2024.",
        "notes": "Multi-hop: verify effective dates across both NDA documents.",
    },
    {
        "question": "Does TechCorp's Remote Work Policy address security requirements consistent with the Information Security Policy?",
        "gold_answer": "The Remote Work Policy requires employees to maintain secure remote connections, consistent with the Information Security Policy.",
        "notes": "Multi-hop: compare remote work security requirements to the information security policy.",
    },
    {
        "question": "Are intellectual property assignment clauses present in both the MSAs and the Contractor Handbook?",
        "gold_answer": "Yes; the MSAs contain IP assignment clauses and the Contractor Handbook also addresses IP ownership.",
        "notes": "Multi-hop: compare IP provisions in contracts and handbook.",
    },
    {
        "question": "Which documents in the corpus were issued by TechCorp Solutions, Inc. as an internal policy rather than an external agreement?",
        "gold_answer": "The Remote Work Policy, Information Security Policy, and Human Resources Policy are internal TechCorp policies.",
        "notes": "Multi-hop: distinguish internal policies from bilateral agreements across all 12 documents.",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks_by_strategy = _load_chunks()

    docs = sorted(PROCESSED_DIR.glob("*.json"))
    candidates: list[dict] = []
    q_counter = 0

    _EXTRACTORS = {
        "contract": _extract_contract,
        "nda": _extract_nda,
        "dpa": _extract_dpa,
        "policy": _extract_policy,
        "handbook": _extract_handbook,
    }

    for doc_path in docs:
        doc = json.loads(doc_path.read_text())
        doc_type = doc.get("doc_type", "unknown")
        full_text = "".join(p["text"] for p in doc["pages"])
        extractor = _EXTRACTORS.get(doc_type)
        if not extractor:
            continue

        facts = extractor(doc, full_text)
        seen_questions: set[str] = set()

        for fact in facts:
            q = fact["question"]
            if q in seen_questions:
                continue
            seen_questions.add(q)

            gold_chunk_ids = _chunks_for_span(
                chunks_by_strategy,
                fact["doc_id"],
                fact["char_start"],
                fact["char_end"],
            )
            if len(gold_chunk_ids) < 3:
                continue  # skip if we can't map to all 3 strategies

            q_counter += 1
            entry = {
                "id": f"q{q_counter:03d}",
                "question": q,
                "gold_answer": fact["answer_text"][:250],
                "gold_chunk_ids": gold_chunk_ids,
                "type": "factual",
                "notes": f"{PENDING_MARKER} | source={fact['doc_id'][:16]} doc_type={fact['doc_type']}",
            }
            candidates.append(entry)

    # Add multi-hop candidates (chunk IDs to be filled by human curator)
    for mh in _MULTI_HOP:
        q_counter += 1
        candidates.append(
            {
                "id": f"q{q_counter:03d}",
                "question": mh["question"],
                "gold_answer": mh["gold_answer"],
                "gold_chunk_ids": {},
                "type": "multi_hop",
                "notes": f"{PENDING_MARKER} | {mh['notes']} — chunk IDs must be filled by curator",
            }
        )

    # Add unanswerable candidates
    for q, notes in zip(_UNANSWERABLE, _UNANSWERABLE_NOTES):
        q_counter += 1
        candidates.append(
            {
                "id": f"q{q_counter:03d}",
                "question": q,
                "gold_answer": "This information is not available in the provided documents.",
                "gold_chunk_ids": {},
                "type": "unanswerable",
                "notes": f"{PENDING_MARKER} | {notes}",
            }
        )

    with OUT_PATH.open("w") as fh:
        for entry in candidates:
            fh.write(json.dumps(entry) + "\n")

    n_factual = sum(1 for c in candidates if c["type"] == "factual")
    n_multi = sum(1 for c in candidates if c["type"] == "multi_hop")
    n_unans = sum(1 for c in candidates if c["type"] == "unanswerable")

    print(f"Proposed {len(candidates)} candidates → {OUT_PATH}")
    print(f"  factual={n_factual}  multi_hop={n_multi}  unanswerable={n_unans}")
    print()
    print("Next steps:")
    print("  1. Review data/eval/golden.proposed.jsonl")
    print("  2. Prune/edit to 75 entries (45 factual / 15 multi_hop / 15 unanswerable)")
    print("  3. Fill gold_chunk_ids for multi_hop entries")
    print("  4. Copy to data/eval/golden.jsonl")
    print("  5. Run: make golden-validate")
    print()
    print("Sample factual candidates:")
    for c in [x for x in candidates if x["type"] == "factual"][:3]:
        print(f"  [{c['id']}] {c['question']}")
        print(f"    A: {c['gold_answer'][:80]}...")
        strats = list(c["gold_chunk_ids"].keys())
        print(f"    strategies mapped: {strats}")


if __name__ == "__main__":
    main()
