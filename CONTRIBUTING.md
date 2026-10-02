# Contributing

This is a personal portfolio project, but I welcome issues, ideas, and pull requests.

## Development setup

    git clone https://github.com/Vickkywhyte/doc-intelligence-platform.git
    cd doc-intelligence-platform
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env       # add OPENROUTER_API_KEY
    make corpus ingest chunk embed index
    pytest tests/ -q           # 378 tests should pass

## Project conventions

- Phases, not features. Read claude_chat/04_ROADMAP.md before adding anything.
- Config over code. Chunk sizes, model names, thresholds all live in configs/*.yaml.
- Interfaces over implementations. New chunkers, embedders, retrievers, or metrics implement the ABC in the corresponding package and register in the registry.
- Test the metric, not the LLM. Unit tests use tests/mocks/llm.py.
- No LLM calls in unit tests. Integration tests are marked @pytest.mark.integration.

See claude_chat/CLAUDE.md for the full operating manual.

## Commit format

`[phase-N] type(scope): summary`

Types: feat, fix, docs, test, refactor, chore, eval.
