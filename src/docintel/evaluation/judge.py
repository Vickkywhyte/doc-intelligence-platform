"""LLM judge wrapper with prompt-version pinning and disk-based cache.

Cache key: sha256(rendered_prompt + "||" + model + "||" + prompt_version)
Cache path: data/cache/judge/{hash}.json

Prompt files are pinned at construction time.  New prompt versions get new
files (e.g. correctness_v2.txt); existing v1 files are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import structlog

from docintel.generation.llm_client import LLMClient
from docintel.utils.config import EvaluationConfig

log = structlog.get_logger()

_PROMPT_DIR = Path(__file__).parent / "judge_prompts"
_DEFAULT_CACHE_DIR = Path("data/cache/judge")


class JudgeClient:
    """Cached LLM judge for correctness, claim decomposition, and entailment."""

    def __init__(
        self,
        llm: LLMClient,
        eval_config: EvaluationConfig,
        cache_dir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._model = eval_config.judge_model
        self._prompt_version = eval_config.judge_prompt_version
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Load and freeze prompt templates at construction time
        self._prompts: dict[str, str] = {}
        for name in ("correctness", "claim_decomposition", "entailment", "refusal_detection"):
            path = _PROMPT_DIR / f"{name}_{self._prompt_version}.txt"
            if path.exists():
                self._prompts[name] = path.read_text()

    # ── Public API ────────────────────────────────────────────────────────────

    def judge_correctness(
        self,
        question: str,
        gold_answer: str,
        answer: str,
    ) -> dict:
        """Return {"score": int 1-5, "rationale": str}."""
        prompt = self._prompts["correctness"].format(
            question=question,
            gold_answer=gold_answer,
            answer=answer,
        )
        result = self._cached_call("correctness", prompt)
        return {"score": int(result.get("score", 1)), "rationale": result.get("rationale", "")}

    def decompose_claims(self, answer: str) -> list[str]:
        """Return a list of atomic factual claims extracted from *answer*."""
        if not answer or not answer.strip():
            return []
        prompt = self._prompts["claim_decomposition"].format(answer=answer)
        result = self._cached_call("claim_decomposition", prompt)
        return list(result.get("claims", []))

    def check_entailment(
        self,
        claim: str,
        chunks: list[tuple[str, str]],
    ) -> dict:
        """Return {"verdict": "yes|partial|no", "chunk_id": str|None}.

        Args:
            claim:  The atomic claim to check.
            chunks: List of (chunk_id, text) pairs to check against.
        """
        chunks_text = "\n".join(f"[{cid}] {text}" for cid, text in chunks)
        prompt = self._prompts["entailment"].format(claim=claim, chunks=chunks_text)
        result = self._cached_call("entailment", prompt)
        verdict = result.get("verdict", "no")
        if verdict not in ("yes", "partial", "no"):
            verdict = "no"
        return {"verdict": verdict, "chunk_id": result.get("chunk_id")}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cache_key(self, rendered_prompt: str) -> str:
        payload = f"{rendered_prompt}||{self._model}||{self._prompt_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> dict | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_cache(self, key: str, result: dict) -> None:
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(result, ensure_ascii=False))
        except OSError as exc:
            log.warning("judge_cache_write_failed", key=key, error=str(exc))

    def _cached_call(self, prompt_name: str, rendered_prompt: str) -> dict:
        key = self._cache_key(rendered_prompt)
        cached = self._load_cache(key)
        if cached is not None:
            log.debug("judge_cache_hit", prompt=prompt_name, key=key[:8])
            return cached

        log.debug("judge_call_start", prompt=prompt_name, model=self._model)
        response = self._llm.generate(
            rendered_prompt,
            temperature=0.0,
            max_tokens=512,
        )

        try:
            # Strip markdown code fences if present
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
        except (json.JSONDecodeError, IndexError) as exc:
            log.warning(
                "judge_json_parse_failed",
                prompt=prompt_name,
                error=str(exc),
                raw=response.text[:200],
            )
            result = {}

        self._save_cache(key, result)
        log.info(
            "judge_call_complete",
            prompt=prompt_name,
            model=self._model,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )
        return result
