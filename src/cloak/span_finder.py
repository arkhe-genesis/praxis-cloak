"""Stage 2 of the decomposed pipeline: LLM span finder.

Asks the local model to list every sensitive entity in the message, in
line-format `SPAN | CATEGORY`. Parses the output deterministically into a list
of {span, category} dicts. Hallucinated spans (not substrings of the original
message) and invalid categories are filtered out at parse time.
"""

import os
import time

import requests

from .models import CATEGORIES

OLLAMA_URL = os.environ.get("CLOAK_OLLAMA_URL", "http://localhost:11434/api/chat")


def _one_call(model: str, messages: list[dict], temperature: float, timeout: int) -> str:
    endpoint = os.getenv("CLOAK_SPAN_ENDPOINT")
    if endpoint:
        resp = requests.post(
            endpoint,
            json={"model": model, "messages": messages, "temperature": temperature,
                  "max_tokens": 256},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False,
              "options": {"temperature": temperature}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_span_finder(
    model: str,
    system_prompt: str,
    raw_message: str,
    timeout: int = 120,
) -> tuple[list[dict], str, int]:
    """Returns (spans, raw_output, elapsed_ms).

    Default backend is Ollama. Set CLOAK_SPAN_ENDPOINT to an OpenAI-compatible
    /v1/chat/completions URL (e.g. an mlx_lm.server).

    CLOAK_SPAN_SAMPLES=N (N>1) enables SELF-CONSISTENCY UNION: sample the finder N
    times at temperature and union the parsed spans. Since substitution is reversible,
    over-detection is cheap and a miss is a leak — so unioning samples is pure recall.
    """
    t0 = time.monotonic()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_message},
    ]
    n = max(1, int(os.getenv("CLOAK_SPAN_SAMPLES", "1")))
    # ENSEMBLE: a comma-separated model list unions spans across models. Different
    # detectors miss different entities, so the union sharply raises recall (leak down),
    # at the cost of more calls + some extra over-detection (reversible, so cheap).
    models = [m.strip() for m in model.split(",") if m.strip()]
    if len(models) == 1 and n == 1:
        raw_output = _one_call(model, messages, 0.0, timeout)
        return parse_span_finder_output(raw_output, raw_message), raw_output, int((time.monotonic() - t0) * 1000)

    seen: set[tuple[str, str]] = set()
    union: list[dict] = []
    outputs: list[str] = []
    for m in models:
        for i in range(n):
            out = _one_call(m, messages, 0.0 if i == 0 else 0.6, timeout)
            outputs.append(out)
            for s in parse_span_finder_output(out, raw_message):
                key = (s["span"], s["category"])
                if key not in seen:
                    seen.add(key)
                    union.append(s)
    return union, "\n---\n".join(outputs), int((time.monotonic() - t0) * 1000)


def parse_span_finder_output(output: str, raw_message: str) -> list[dict]:
    """Parse line-format span finder output into structured spans.

    Filters out hallucinated spans (not in raw_message) and unknown categories.
    """
    spans: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if "|" not in line:
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        span = parts[0].strip()
        category = parts[1].strip().lower()
        if not span or not category:
            continue
        if span not in raw_message:
            # Hallucinated span — model named something that isn't actually there
            continue
        if category not in CATEGORIES:
            continue
        key = (span, category)
        if key in seen:
            continue
        seen.add(key)
        spans.append({"span": span, "category": category})
    return spans
