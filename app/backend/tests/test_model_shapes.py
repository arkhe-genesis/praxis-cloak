"""Offline request-shape regression tests for the cloud model matrix.

No network, no keys: we stub the provider SDK clients and assert the EXACT request kwargs
each (model, effort) combo produces. This pins the API surface so a rename / wrong param /
bad budget math fails here instead of in production. Catalog-driven via `enumerate_cases`,
so a new model is auto-covered; a new *provider* must add a shape expectation below (the
`test_every_provider_has_expectation` guard fails loudly until it does).

Run: cd app && .venv/bin/python -m pytest backend/tests -q
"""

from __future__ import annotations

import pytest

from backend import providers
from backend.config import Config
from backend.model_check import Case, enumerate_cases
from backend.transport import DirectTransport, OpenAITransport

# Fake keys so every provider is "available" and we exercise the whole matrix.
CFG = Config(
    local_model="x",
    relevance_model="praxis/relevance-3b",
    cloud_model="claude-sonnet-4-6",
    span_prompt_name="s",
    anthropic_api_key="sk-ant-fake",
    openai_api_key="sk-openai-fake",
    host="127.0.0.1",
    port=8765,
    max_tokens=1024,
)
MSGS = [{"role": "user", "content": "ping"}]


# -- fake SDK clients that record the request kwargs ---------------------------

class _StreamCtx:
    def __init__(self):
        self.text_stream = iter(["ok"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeAnthropic:
    def __init__(self, sink: dict):
        self._sink = sink
        self.messages = self  # so .messages.stream works

    def stream(self, **kwargs):
        self._sink.update(kwargs)
        return _StreamCtx()


class _Event:
    type = "response.output_text.delta"
    delta = "ok"


class _FakeOpenAI:
    def __init__(self, sink: dict):
        self._sink = sink
        self.responses = self  # so .responses.create works

    def create(self, **kwargs):
        self._sink.update(kwargs)
        return iter([_Event()])


def _capture(case: Case) -> dict:
    """Build the transport for `case`, inject a recording fake client, run the stream, and
    return the request kwargs that reached the SDK."""
    transport = providers.build_transport(case.model, case.effort or None, CFG)
    sink: dict = {}
    if isinstance(transport, OpenAITransport):
        transport._client = _FakeOpenAI(sink)
    elif isinstance(transport, DirectTransport):
        transport._client = _FakeAnthropic(sink)
    else:  # pragma: no cover - new transport types should add a fake above
        pytest.fail(f"no fake client for transport {type(transport).__name__}")
    list(transport.stream(MSGS))  # force the generator so kwargs are recorded
    return sink


# -- per-provider expectations (a new provider MUST add one here) --------------

def _expect_anthropic(case: Case, kw: dict) -> None:
    assert kw["model"] == case.model
    assert kw["messages"] == MSGS
    spec = next(m for m in providers.CATALOG if m.id == case.model)
    if spec.thinking_mode == "adaptive" and case.effort:
        # newer API: adaptive thinking + output_config.effort (sent via extra_body)
        assert kw.get("extra_body") == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": case.effort},
        }
        assert "thinking" not in kw  # not the typed enabled-budget param
        assert kw["max_tokens"] == DirectTransport._ADAPTIVE_MAX_TOKENS
    elif spec.thinking_mode == "budget":
        budget = DirectTransport._BUDGET.get(case.effort.lower()) if case.effort else None
        if budget:
            assert kw.get("thinking") == {"type": "enabled", "budget_tokens": budget}
            assert "extra_body" not in kw
            assert kw["max_tokens"] == budget + max(CFG.max_tokens, 4096)
        else:
            assert "thinking" not in kw and "extra_body" not in kw
            assert kw["max_tokens"] == CFG.max_tokens
    else:
        assert "thinking" not in kw and "extra_body" not in kw
        assert kw["max_tokens"] == CFG.max_tokens


def _expect_openai(case: Case, kw: dict) -> None:
    assert kw["model"] == case.model
    assert kw["input"] == MSGS  # Responses API, not chat.completions
    assert "messages" not in kw
    assert kw.get("stream") is True
    if case.effort:
        assert kw.get("reasoning") == {"effort": case.effort}
        assert kw["max_output_tokens"] == OpenAITransport._REASONING_MAX_OUTPUT
    else:
        assert "reasoning" not in kw
        assert kw["max_output_tokens"] == CFG.max_tokens


EXPECT = {"anthropic": _expect_anthropic, "openai": _expect_openai}


# -- tests --------------------------------------------------------------------

def test_every_provider_has_expectation():
    """Forcing function: adding a provider to the catalog without shape coverage fails."""
    for p in providers.PROVIDERS:
        assert p.id in EXPECT, f"add a shape expectation for provider '{p.id}'"


def test_cases_cover_all_models():
    covered = {c.model for c in enumerate_cases(CFG)}
    assert covered == {m.id for m in providers.CATALOG}


def test_quick_is_one_case_per_model():
    quick = enumerate_cases(CFG, quick=True)
    per_model = [c.model for c in quick]
    assert sorted(per_model) == sorted(m.id for m in providers.CATALOG)


@pytest.mark.parametrize(
    "case", enumerate_cases(CFG), ids=lambda c: f"{c.model}::{c.effort or 'none'}"
)
def test_request_shape(case: Case):
    kw = _capture(case)
    EXPECT[case.provider](case, kw)
