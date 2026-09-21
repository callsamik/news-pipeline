"""LLM client Protocol — production impl is multiprovider-llm.Client."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMCompletion(Protocol):
    """Minimal completion surface (matches multiprovider_llm.CompletionResult.text)."""

    @property
    def text(self) -> str: ...


@runtime_checkable
class LLMClient(Protocol):
    """Protocol compatible with ``multiprovider_llm.Client.complete``.

    ``news-pipeline`` accepts this Protocol. ``multiprovider-llm.Client`` is the
    intended production implementation; the library does not import it in core.
    """

    def complete(
        self,
        *,
        prompt: str | None = None,
        messages: Sequence[Any] | None = None,
        tier: str | None = None,
        provider_chain: Sequence[str] | None = None,
        response_format: Literal["text", "json"] = "text",
        json_schema: Mapping[str, Any] | None = None,
        freshness_required: bool = False,
        timeout_s: float | None = None,
        include_raw: bool = False,
        max_tokens: int | None = None,
        on_auth_failure: Literal["stop", "continue"] = "stop",
    ) -> LLMCompletion: ...
