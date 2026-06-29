"""The one place the app reaches into the scrub engine.

The scrub engine ships as the installable `cloak` library (extracted from the eval
harness). We treat it as a read-only dependency and re-export exactly the callables the app
needs. Nothing else in the app imports `cloak` directly, so if the library's layout
moves, only this file changes.
"""

from __future__ import annotations

from functools import lru_cache

from cloak import (  # noqa: F401
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_MODEL,
    Profile,
    Replacement,
    fast_scrub,
    load_span_prompt,
    rehydrate,
)

__all__ = [
    "fast_scrub",
    "rehydrate",
    "Replacement",
    "Profile",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_CLOUD_MODEL",
    "span_prompt",
]


@lru_cache(maxsize=None)
def span_prompt(name: str = "span_finder_v0_5.txt") -> str:
    """Load a span-finder prompt by filename from the library's bundled prompts."""
    return load_span_prompt(name)
