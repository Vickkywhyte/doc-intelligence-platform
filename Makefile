.PHONY: help install check-config corpus ingest test clean clean-cache index query-smoke rag-smoke golden-seed golden-validate golden-rewrite golden-dedupe eval-single test-metrics smoke-eval

CONFIG ?= configs/experiment_matrix.yaml
PYTHON  ?= python3.11

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install package and dev deps
	pip install -e ".[dev]"

# ── Phase 1 ──────────────────────────────────────────────────────────────────

check-config: ## Print resolved config as JSON + config_hash; exits 1 on bad config
	$(PYTHON) -m docintel.utils.config $(CONFIG)

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run the full test suite
	pytest -q tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove Python build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ dist/ build/

clean-cache: ## Remove embedding and judge caches (forces re-computation)
	rm -rf data/cache/embeddings/ data/cache/judge/
	@echo "Caches cleared."

# ── Placeholder targets for future phases ─────────────────────────────────────
# (stubs only; implemented when those phases are active)

corpus: ## Generate synthetic corpus of 12 PDFs to data/raw/
	mkdir -p data/raw
	$(PYTHON) scripts/00_synthesize_corpus.py

ingest: ## Ingest data/raw/* → data/processed/{doc_id}.json
	mkdir -p data/processed
	$(PYTHON) scripts/01_ingest.py

chunk: ## Chunk data/processed/* → data/chunks/{strategy}/{doc_id}.jsonl
	mkdir -p data/chunks
	$(PYTHON) scripts/02_chunk.py

embed: ## Embed data/chunks/* → data/cache/embeddings/ (run twice to verify cache)
	mkdir -p data/cache/embeddings
	$(PYTHON) scripts/03_embed.py

index: ## Embed chunks + upsert into data/indexes/{strategy}_{embedder}/ (6 dirs)
	mkdir -p data/indexes
	$(PYTHON) scripts/03b_embed_and_index.py

query-smoke: ## Smoke-test retrieval with 3 seed questions across all 6 indexes
	$(PYTHON) scripts/03c_query_smoke.py

rag-smoke: ## Smoke-test RAG pipeline: 3 in-corpus (citations) + 1 OOD (refusal)
	$(PYTHON) scripts/04_rag_smoke.py

golden-seed: ## Propose golden candidates to data/eval/golden.proposed.jsonl (PENDING REVIEW)
	mkdir -p data/eval
	$(PYTHON) scripts/05_seed_golden.py

golden-validate: ## Validate data/eval/golden.jsonl schema, composition, and chunk IDs
	$(PYTHON) scripts/06_validate_golden.py

golden-rewrite: ## Rewrite fragmented gold answers via gpt-4o-mini → golden.rewritten.jsonl
	$(PYTHON) scripts/07_rewrite_golden.py

golden-dedupe: ## Doc-qualify duplicate questions → golden.deduped.jsonl (no LLM)
	$(PYTHON) scripts/08_dedupe_golden.py

eval-single: ## Run one-config evaluation over the full golden set → results/runs/{run_id}.json
	mkdir -p results/runs
	$(PYTHON) -m docintel.evaluation.evaluator --config $(CONFIG)

test-metrics: ## Run only metric unit tests (fast, no LLM)
	pytest tests/test_metrics.py -v

smoke-eval: ## Smoke-test pipeline on 6 golden items (~$0.02, needs real indexes)
	$(PYTHON) scripts/09_smoke_eval.py --config $(CONFIG)

eval-matrix:
	@echo "Phase 9: not yet implemented"

report:
	@echo "Phase 9: not yet implemented"

serve:
	@echo "Phase 10: not yet implemented"
