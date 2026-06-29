"""Cloud model catalog + provider selection.

The app talks to one cloud model at a time, chosen in the composer dropdown. Each model
belongs to a provider (anthropic | openai); a provider is "available" when its API key is
present in the environment. Most models also expose a *reasoning* control — Anthropic calls
it effort/thinking, OpenAI calls it intelligence — surfaced as a nested submenu in the
picker and applied per-send by the transport.

This module is the single source of truth for the catalog and for turning a (model, effort)
pair into a built transport. Keys are bring-your-own (BYOK): build_transport() takes the
per-request anthropic/openai keys and falls back to the env-configured keys when those are
absent. A missing key never fails the build — it surfaces only when a cloud call is made.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .transport import ChatTransport, DirectTransport, OpenAITransport


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    key_env: str  # the environment variable that makes this provider available


@dataclass(frozen=True)
class EffortOption:
    value: str  # provider-specific value the transport interprets
    label: str  # what the user sees in the submenu


@dataclass(frozen=True)
class Reasoning:
    label: str  # submenu title / trigger suffix kind: "Effort" | "Intelligence" | "Thinking"
    default: str  # default option value
    options: tuple[EffortOption, ...]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    provider: str
    reasoning: Reasoning | None = None
    thinking_mode: str | None = None  # anthropic only: "adaptive" | "budget" (see transport)


PROVIDERS: list[ProviderSpec] = [
    ProviderSpec("anthropic", "Anthropic", "ANTHROPIC_API_KEY"),
    ProviderSpec("openai", "OpenAI", "OPENAI_API_KEY"),
]

# Anthropic reasoning (verified live, see docs/reference/cloud-models.md). Opus/Sonnet use
# the adaptive API where effort values ARE the API enum (Opus: low/medium/high/xhigh/max;
# Sonnet drops xhigh). Haiku uses the older budget API as a Standard/Extended toggle.
_OPUS_EFFORT = Reasoning(
    "Effort",
    "high",
    (
        EffortOption("low", "Low"),
        EffortOption("medium", "Medium"),
        EffortOption("high", "High"),
        EffortOption("xhigh", "Extra"),
        EffortOption("max", "Max"),
    ),
)
_SONNET_EFFORT = Reasoning(
    "Effort",
    "high",
    (
        EffortOption("low", "Low"),
        EffortOption("medium", "Medium"),
        EffortOption("high", "High"),
        EffortOption("max", "Max"),
    ),
)
_HAIKU_THINKING = Reasoning(
    "Thinking",
    "standard",
    (EffortOption("standard", "Standard"), EffortOption("extended", "Extended")),
)
# OpenAI GPT-5.x "intelligence" → Responses API reasoning.effort. gpt-5.5 defaults to medium.
_GPT5_INTELLIGENCE = Reasoning(
    "Intelligence",
    "medium",
    (
        EffortOption("none", "Instant"),  # documented latency tier; swap to "minimal"/"low" if preferred
        EffortOption("medium", "Medium"),
        EffortOption("high", "High"),
        EffortOption("xhigh", "Extra High"),
    ),
)

CATALOG: list[ModelSpec] = [
    ModelSpec("claude-opus-4-8", "Claude Opus 4.8", "anthropic", _OPUS_EFFORT, "adaptive"),
    ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6", "anthropic", _SONNET_EFFORT, "adaptive"),
    ModelSpec("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "anthropic", _HAIKU_THINKING, "budget"),
    ModelSpec("gpt-5.5", "GPT-5.5", "openai", _GPT5_INTELLIGENCE),
    ModelSpec("gpt-5.4", "GPT-5.4", "openai", _GPT5_INTELLIGENCE),
    ModelSpec("gpt-5.4-mini", "GPT-5.4 mini", "openai", _GPT5_INTELLIGENCE),
]

_PROVIDERS_BY_ID = {p.id: p for p in PROVIDERS}
_MODELS_BY_ID = {m.id: m for m in CATALOG}


def _key_for(provider: str, cfg: Config) -> str | None:
    if provider == "anthropic":
        return cfg.anthropic_api_key
    if provider == "openai":
        return cfg.openai_api_key
    return None


def provider_available(provider: str, cfg: Config) -> bool:
    return bool(_key_for(provider, cfg))


def default_model(cfg: Config) -> str:
    """The configured cloud model if its provider key is present, else the first model
    whose provider is available, else the configured model anyway (so the UI still shows
    something and the error surfaces only on send)."""
    configured = _MODELS_BY_ID.get(cfg.cloud_model)
    if configured and provider_available(configured.provider, cfg):
        return configured.id
    for m in CATALOG:
        if provider_available(m.provider, cfg):
            return m.id
    return cfg.cloud_model


def default_effort(model_id: str) -> str | None:
    spec = _MODELS_BY_ID.get(model_id)
    return spec.reasoning.default if (spec and spec.reasoning) else None


def is_known_model(model_id: str | None) -> bool:
    return model_id in _MODELS_BY_ID


def effort_values(model_id: str) -> tuple[str, ...]:
    """Valid effort option values for a model (() if it has no reasoning control)."""
    spec = _MODELS_BY_ID.get(model_id)
    return tuple(o.value for o in spec.reasoning.options) if (spec and spec.reasoning) else ()


def catalog(cfg: Config) -> dict:
    """Everything the composer's model picker needs in one call."""
    return {
        "models": [
            {
                "id": m.id,
                "label": m.label,
                "provider": m.provider,
                "available": provider_available(m.provider, cfg),
                "reasoning": None
                if m.reasoning is None
                else {
                    "label": m.reasoning.label,
                    "default": m.reasoning.default,
                    "options": [{"value": o.value, "label": o.label} for o in m.reasoning.options],
                },
            }
            for m in CATALOG
        ],
        "providers": [
            {
                "id": p.id,
                "label": p.label,
                "available": provider_available(p.id, cfg),
                "key_env": p.key_env,
            }
            for p in PROVIDERS
        ],
        "default": default_model(cfg),
    }


def build_transport(
    model_id: str | None,
    effort: str | None,
    cfg: Config,
    anthropic_key: str | None = None,
    openai_key: str | None = None,
) -> ChatTransport:
    """Build the transport for `model_id` + `effort` (falling back to defaults). BYOK: the
    per-request `anthropic_key`/`openai_key` win, falling back to the env-configured keys.
    A missing key never fails the build — it raises a clear error only at stream time when
    a cloud call is actually made."""
    mid = model_id or default_model(cfg)
    spec = _MODELS_BY_ID.get(mid)
    eff = effort if effort is not None else default_effort(mid)
    anthropic = anthropic_key or cfg.anthropic_api_key
    openai = openai_key or cfg.openai_api_key
    if spec is None:
        # Unknown id (e.g. a CLOAK_CLOUD_MODEL override outside the catalog): assume
        # Anthropic-style, no reasoning.
        return DirectTransport(model=mid, api_key=anthropic, max_tokens=cfg.max_tokens)
    if spec.provider == "openai":
        return OpenAITransport(
            model=spec.id, api_key=openai, max_tokens=cfg.max_tokens, effort=eff
        )
    return DirectTransport(
        model=spec.id,
        api_key=anthropic,
        max_tokens=cfg.max_tokens,
        effort=eff,
        thinking_mode=spec.thinking_mode,
    )
