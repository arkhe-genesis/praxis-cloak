"""The boundary between the local app and the cloud model.

This open-source build is bring-your-own-key (BYOK): each transport is constructed with
the user's provider key and calls the provider's API directly. The transports only know
about the ChatTransport interface and the scrubbed message list, so the scrub, the
conversation logic, and the frontend stay decoupled from credential handling. Which
transport to build (provider/model/effort selection) lives in `providers.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class ChatTransport(ABC):
    """Streams an assistant reply for an already-scrubbed message list."""

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield text deltas for the reply. `messages` is the scrubbed conversation."""
        raise NotImplementedError


class DirectTransport(ChatTransport):
    """Anthropic, called directly with the user's BYOK key, streaming.

    The *content* is scrubbed locally before it ever leaves the machine; the honest
    BYOK consequence is that the provider still sees the user's own account. An optional
    `base_url` lets a self-hoster point at an Anthropic-compatible endpoint.

    Reasoning differs by model generation (verified live, see docs/reference/cloud-models.md):
      * thinking_mode="adaptive" (Opus 4.8, Sonnet 4.6): newer API — `thinking.type=adaptive`
        + `output_config.effort` (low/medium/high/[xhigh]/max), passed verbatim.
      * thinking_mode="budget" (Haiku 4.5): older API — `thinking.type=enabled` + a
        budget_tokens (effort "extended"); "standard"/None = no thinking.
    The SDK's `text_stream` yields only the visible answer (thinking deltas excluded), so
    the rest of the pipeline is unchanged regardless of mode.
    """

    # budget-mode efforts -> extended-thinking budget_tokens (Haiku's "extended" on-state).
    _BUDGET = {"extended": 8000}
    # adaptive models share one max_tokens between thinking + answer; keep it generous.
    _ADAPTIVE_MAX_TOKENS = 24000

    def __init__(
        self,
        model: str,
        api_key: str | None,
        max_tokens: int = 1024,
        effort: str | None = None,
        thinking_mode: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking_mode = thinking_mode
        self.base_url = base_url  # optional custom Anthropic-compatible endpoint; None = default
        self._api_key = api_key
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "No Anthropic API key — add your key in Settings (it stays on your "
                    "device), or pick a model whose provider key you've set."
                )
            from anthropic import Anthropic

            kwargs = {"api_key": self._api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def stream(self, messages: list[dict]) -> Iterator[str]:
        client = self._client_or_raise()
        kwargs: dict = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if self.thinking_mode == "adaptive" and self.effort:
            kwargs["max_tokens"] = self._ADAPTIVE_MAX_TOKENS
            kwargs["extra_body"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": self.effort},
            }
        elif self.thinking_mode == "budget":
            budget = self._BUDGET.get((self.effort or "").lower())
            if budget:
                # max_tokens must exceed the thinking budget; leave room for the answer too.
                kwargs["max_tokens"] = budget + max(self.max_tokens, 4096)
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream


class OpenAITransport(ChatTransport):
    """OpenAI, called directly with the user's BYOK key, streaming via the Responses API.

    Reasoning models (GPT-5.x) work best with the Responses API + `reasoning.effort`
    (Instant/Medium/High/Extra High in the UI -> minimal/medium/high/xhigh). The scrubbed
    message list ({role, content}) is accepted as `input` unchanged. Same BYOK caveat as
    DirectTransport: content is scrubbed locally, but the provider call is on the user's
    own account.
    """

    # Reasoning spends tokens before any visible output; reserve generous headroom so a
    # reply isn't cut off mid-thought (OpenAI suggests ≥25k). Only applied when reasoning.
    _REASONING_MAX_OUTPUT = 32000

    def __init__(
        self,
        model: str,
        api_key: str | None,
        max_tokens: int = 1024,
        effort: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.base_url = base_url  # optional custom OpenAI-compatible endpoint; None = default
        self._api_key = api_key
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "No OpenAI API key — add your key in Settings (it stays on your "
                    "device), or pick a model whose provider key you've set."
                )
            from openai import OpenAI

            kwargs = {"api_key": self._api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def stream(self, messages: list[dict]) -> Iterator[str]:
        client = self._client_or_raise()
        kwargs: dict = {"model": self.model, "input": messages, "stream": True}
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
            kwargs["max_output_tokens"] = self._REASONING_MAX_OUTPUT
        else:
            kwargs["max_output_tokens"] = self.max_tokens
        stream = client.responses.create(**kwargs)
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "response.output_text.delta":
                yield event.delta
            elif etype == "error":
                raise RuntimeError(getattr(event, "message", "OpenAI stream error"))
