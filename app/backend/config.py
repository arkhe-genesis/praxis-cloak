"""Runtime configuration, all overridable by environment variable.

Defaults mirror the proven chat_server config (the harness DEFAULT_* models +
span_finder_v0_5). Cloud keys are bring-your-own (BYOK): the frontend supplies an
Anthropic and/or OpenAI key per request; the environment values (ANTHROPIC_API_KEY /
OPENAI_API_KEY) act as fallbacks for a self-hosted operator who would rather set them
once at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import harness_bridge as hb

# The fine-tuned on-device span-finder finds the PII to scrub. v7 fixes the first-person
# framing bias v4 had (it found "I live in Paris" but missed "weather in Paris" / "cover
# letter to Google"): user-PII recall on task-framed prompts 45%->71% with precision held
# (fuzz harness, 2026-06-17; docs/explorations/2026-06-17-detection-fuzzing-and-framing-bias.md).
# Default is the q4_k_m build: same user-PII recall as q8 (71%) at v4's 1.9GB size (q8 only adds
# ~6pp on public task-subjects the relevance judge keeps anyway). `-v7:latest` is the q8/3.3GB
# variant if you want the marginal public-subject recall. Overridable via CLOAK_LOCAL_MODEL.
_DEFAULT_LOCAL_MODEL = "praxis/spanfinder-3b"

# The fine-tuned on-device relevance judge makes the keep-vs-scrub call for locations/orgs (the
# anti-lobotomy decision). Without it, fast_scrub falls back to the keyword keep-gate, which
# over-scrubs load-bearing locations (e2e: 52% kept vs the judge's 93%). Default-on; the winner
# of the 2026-06-17 model+eval work (FT-3B-v2; docs/explorations/2026-06-17-relevance-customer-ready.md).
# Override via CLOAK_RELEVANCE_MODEL; set it to "" to disable and use the keyword gate.
_DEFAULT_RELEVANCE_MODEL = "praxis/relevance-3b"


@dataclass(frozen=True)
class Config:
    local_model: str
    relevance_model: str
    cloud_model: str
    span_prompt_name: str
    anthropic_api_key: str | None
    openai_api_key: str | None
    host: str
    port: int
    max_tokens: int


def load_config() -> Config:
    return Config(
        local_model=os.environ.get("CLOAK_LOCAL_MODEL", _DEFAULT_LOCAL_MODEL),
        relevance_model=os.environ.get("CLOAK_RELEVANCE_MODEL", _DEFAULT_RELEVANCE_MODEL),
        cloud_model=os.environ.get("CLOAK_CLOUD_MODEL", hb.DEFAULT_CLOUD_MODEL),
        span_prompt_name=os.environ.get("CLOAK_SPAN_PROMPT", "span_finder_v0_5.txt"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        host=os.environ.get("CLOAK_HOST", "127.0.0.1"),
        port=int(os.environ.get("CLOAK_PORT", "8765")),
        max_tokens=int(os.environ.get("CLOAK_MAX_TOKENS", "1024")),
    )
