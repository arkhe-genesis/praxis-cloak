"""Cloak: fast, reversible PII scrubbing + rehydration for LLM queries.

Core functions
--------------
- ``scrub_message(message, ...)`` — convenience wrapper around :func:`fast_scrub`
  that defaults the span-finder prompt to the bundled ``span_finder_v0_5.txt``.
- ``fast_scrub(message, span_finder_model, span_prompt, ...)`` — the low-level scrub:
  replaces every identifying entity with a realistic, reversible fake.
- ``rehydrate(cloud_response, replacements, ...)`` — restore original entities in the
  cloud model's response.

Data classes
------------
- :class:`Replacement` — ``(original, replacement, category)``.
- :class:`RehydrationResult` — rehydrated text + per-replacement events.
- :class:`Profile` — a user's known-entity profile for deterministic matching.

Configuration (environment variables)
-------------------------------------
- ``CLOAK_OLLAMA_URL`` (default ``http://localhost:11434/api/chat``) — local backend.
- ``CLOAK_SPAN_ENDPOINT`` — OpenAI-compatible override for the span finder.
- ``CLOAK_RELEVANCE_ENDPOINT`` — endpoint override for the relevance judge.
- ``CLOAK_RELEVANCE_MODEL`` — opt-in fine-tuned relevance judge model.
"""

from __future__ import annotations

from pathlib import Path

from .combine import fast_scrub
from .models import RehydrationResult, Replacement
from .pipeline import DEFAULT_CLOUD_MODEL, DEFAULT_LOCAL_MODEL
from .profile_match import Profile
from .rehydrate import rehydrate

__version__ = "0.1.0"

__all__ = [
    "fast_scrub",
    "scrub_message",
    "rehydrate",
    "load_span_prompt",
    "Replacement",
    "RehydrationResult",
    "Profile",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_CLOUD_MODEL",
    "__version__",
]

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_span_prompt(name: str = "span_finder_v0_5.txt") -> str:
    """Return the text of a span-finder system prompt bundled with the package.

    Prompts live in the package's ``prompts/`` directory (shipped as package data),
    so this works for both editable and wheel installs.
    """
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def scrub_message(
    message: str,
    span_finder_model: str = DEFAULT_LOCAL_MODEL,
    span_prompt: str | None = None,
    existing: dict | None = None,
    profile: Profile | None = None,
    relevance_model: str | None = None,
) -> tuple[str, list[Replacement], dict]:
    """Scrub a message, replacing identifiable entities with realistic fakes.

    A thin convenience wrapper around :func:`fast_scrub` that loads the default,
    bundled span-finder prompt when ``span_prompt`` is not supplied.

    Args:
        message: The user's input message.
        span_finder_model: Local model name for entity detection (Ollama tag).
        span_prompt: System prompt for the span finder; defaults to the bundled
            ``span_finder_v0_5.txt`` when ``None``.
        existing: Prior turn's substitution map ``{real: (fake, category)}`` for
            stable substitution across turns.
        profile: A :class:`Profile` of the user's known entities for deterministic
            matching (optional).
        relevance_model: Fine-tuned model name for the question-aware keep-gate
            (optional; pass ``""`` to disable).

    Returns:
        ``(protected_message, replacements, updated_mapping)`` where
        ``updated_mapping`` is ``{original: (fake, category)}`` for rehydration.
    """
    if span_prompt is None:
        span_prompt = load_span_prompt()
    return fast_scrub(
        message,
        span_finder_model,
        span_prompt,
        existing=existing,
        profile=profile,
        relevance_model=relevance_model,
    )
