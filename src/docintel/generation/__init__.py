"""Generation package — LLM client, prompt templates, citation parser."""

from docintel.generation.citation_parser import parse_citations
from docintel.generation.llm_client import LLMClient, OpenRouterClient
from docintel.generation.models import Citation, LLMResponse, RAGResponse
from docintel.generation.prompt_templates import (
    REFUSAL_PHRASES,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_refusal,
)

__all__ = [
    "LLMClient",
    "OpenRouterClient",
    "Citation",
    "LLMResponse",
    "RAGResponse",
    "SYSTEM_PROMPT",
    "REFUSAL_PHRASES",
    "build_user_prompt",
    "is_refusal",
    "parse_citations",
]
