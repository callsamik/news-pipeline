"""Optional LLM summarization — Protocol-based; no provider implementations."""

from __future__ import annotations

from news_pipeline.llm.client_protocol import LLMClient, LLMCompletion
from news_pipeline.llm.summarize_batch import summarize_items_llm

__all__ = [
    "LLMClient",
    "LLMCompletion",
    "summarize_items_llm",
]
