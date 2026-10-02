"""
Deduplicate golden.rewritten.jsonl by doc-qualifying repeated question text.

Reads  : data/eval/golden.rewritten.jsonl
Writes : data/eval/golden.deduped.jsonl

Does NOT call any LLM.
Does NOT modify golden.jsonl or golden.rewritten.jsonl.
Does NOT alter gold_chunk_ids.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
REWRITTEN = ROOT / "data" / "eval" / "golden.rewritten.jsonl"
PROCESSED = ROOT / "data" / "processed"
OUTPUT    = ROOT / "data" / "eval" / "golden.deduped.jsonl"

# ── filename → human label ───────────────────────────────────────────────────

# Pattern: <doctype>_<company>_<year>.pdf  or  <doctype>_<word>_<year>.pdf
# Multi-word doc-types handled by checking the whole stem.
_SPECIAL: dict[str, str] = {
    "policy_hr":         "HR Policy",
    "policy_remote_work": "Remote Work Policy",
    "policy_security":   "Security Policy",
    "handbook_employee": "Employee Handbook",
    "handbook_contractor": "Contractor Handbook",
}


def _label_from_filename(filename: str) -> str:
    """Return a short human-readable label for a source document filename."""
    stem = Path(filename).stem.lower()           # e.g. "dpa_acme_2024"
    # strip trailing _YYYY
    stem_no_year = re.sub(r"_\d{4}$", "", stem)  # e.g. "dpa_acme"

    # check special overrides first
    for key, label in _SPECIAL.items():
        if stem_no_year == key or stem_no_year.startswith(key):
            return label

    # generic: <doctype>_<company…>
    parts = stem_no_year.split("_", 1)           # ["dpa", "acme"] or ["msa", "globex"]
    if len(parts) == 2:
        doc_type, company = parts
        company_cap = company.replace("_", " ").title()
        doc_type_upper = doc_type.upper()
        return f"{company_cap} {doc_type_upper}"

    # fallback: use the raw filename
    return filename


def _load_doc_filename(doc_id: str) -> str:
    path = PROCESSED / f"{doc_id}.json"
    try:
        data = json.loads(path.read_text())
        return data.get("filename") or data.get("source_file") or data.get("file") or path.name
    except FileNotFoundError:
        return doc_id


def _doc_id_from_entry(entry: dict) -> str:
    """Extract doc_id from gold_chunk_ids.fixed[0]."""
    fixed_ids = entry.get("gold_chunk_ids", {}).get("fixed", [])
    if not fixed_ids:
        return ""
    return fixed_ids[0].split(":")[0]


# ── question rewriting ────────────────────────────────────────────────────────

def _qualify_question(question: str, label: str) -> str:
    """
    Insert a doc-specific qualifier into the question text.

    Strategy (in order):
    1. Replace generic doc-type references  ("the Data Processing Agreement"
       → "the Acme DPA") when we can match a known phrase.
    2. Append "… (in the <label>?)" if no natural insertion point found.
    """
    # Mapping of generic phrases we know appear in the corpus → short type abbrev
    _PHRASES: list[tuple[str, str]] = [
        (r"the Data Processing Agreement",  "DPA"),
        (r"the Master Services Agreement",  "MSA"),
        (r"the Non-Disclosure Agreement",   "NDA"),
        (r"the MSA",                        "MSA"),
        (r"the DPA",                        "DPA"),
        (r"the NDA",                        "NDA"),
        (r"the Employee Handbook",          "Employee Handbook"),
        (r"the Contractor Handbook",        "Contractor Handbook"),
        (r"the HR Policy",                  "HR Policy"),
        (r"the Remote Work Policy",         "Remote Work Policy"),
        (r"the Security Policy",            "Security Policy"),
    ]

    for phrase, abbrev in _PHRASES:
        if re.search(phrase, question, flags=re.IGNORECASE):
            # Replace "the <doc>" with "the <label>"
            new_q = re.sub(phrase, f"the {label}", question, flags=re.IGNORECASE, count=1)
            if new_q != question:
                return new_q

    # Fallback: append qualifier before the trailing "?"
    if question.rstrip().endswith("?"):
        return question.rstrip()[:-1] + f" (in the {label})?"
    return f"{question} (in the {label})"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    entries: list[dict] = []
    with REWRITTEN.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {REWRITTEN.name}")

    # group by question text
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[e["question"]].append(i)

    duplicate_groups = {q: idxs for q, idxs in groups.items() if len(idxs) > 1}
    print(f"Unique questions  : {len(groups)}")
    print(f"Duplicate groups  : {len(duplicate_groups)}")
    print()

    transformations: list[tuple[str, str]] = []
    result = [dict(e) for e in entries]   # shallow-copy all entries

    for question, idxs in duplicate_groups.items():
        for idx in idxs:
            doc_id   = _doc_id_from_entry(entries[idx])
            filename = _load_doc_filename(doc_id)
            label    = _label_from_filename(filename)
            new_q    = _qualify_question(question, label)
            result[idx]["question"] = new_q
            transformations.append((question, new_q))

    # ── write output ──────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as fh:
        for entry in result:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(result)} entries to {OUTPUT.name}")

    # ── verify uniqueness ─────────────────────────────────────────────────────
    final_questions = [e["question"] for e in result]
    unique_final    = set(final_questions)
    if len(unique_final) != len(result):
        remaining_dups = {
            q: final_questions.count(q)
            for q in unique_final
            if final_questions.count(q) > 1
        }
        print()
        print("WARNING: uniqueness check FAILED — still-duplicate questions:")
        for q, c in remaining_dups.items():
            print(f"  [{c}x] {q}")
        sys.exit(1)
    else:
        print(f"Uniqueness check  : PASSED ({len(unique_final)} unique == {len(result)} total)")

    # ── preview first 15 transformations ─────────────────────────────────────
    print()
    print("First 15 question transformations (old → new):")
    for old, new in transformations[:15]:
        print(f"  BEFORE: {old}")
        print(f"  AFTER : {new}")
        print()


if __name__ == "__main__":
    main()
