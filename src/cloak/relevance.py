"""Question-aware relevance judge (keep vs scrub) — the fine-tuned on-device model that replaces
the keyword keep-gate for locations / orgs.

INTERIM v2 build (location over-scrub ~11% on CAPID; not customer-grade — see
docs/explorations/2026-06-16-relevance-finetune-rounds.md). Opt-in via
CLOAK_RELEVANCE_MODEL=praxis/relevance-3b. One Ollama call per message: given the
message + the detected spans, returns {span: keep?}. On ANY failure (Ollama down, timeout,
unparsable) returns {} so the caller falls back to the deterministic keyword keep-gate.
"""

import os
import re

import requests

OLLAMA_URL = os.environ.get(
    "CLOAK_RELEVANCE_ENDPOINT",
    os.environ.get("CLOAK_OLLAMA_URL", "http://localhost:11434/api/chat"),
)

# train == serve: byte-identical to build_relevance_data.SYSTEM / eval_relevance_local.SYSTEM.
SYSTEM = ("You decide which personal details in a message are NEEDED to answer the user's question "
          "well. A detail is RELEVANT only if answering the question genuinely depends on that "
          "specific detail (e.g. a location when the answer depends on local law or availability). "
          "It is NOT relevant if it is incidental background the answer does not need.")

_SENT = re.compile(r"[^.!?\n]*[.!?]")


def _split(message: str) -> tuple[str, str]:
    """context = the whole message; question = the last sentence (the likely ask), or the last
    '?'-sentence if there is one. A single message carries both; this is a known interim
    heuristic (a cleaner context/question split is on the near-perfect agenda)."""
    sents = [s.strip() for s in _SENT.findall(message) if s.strip()]
    question = next((s for s in reversed(sents) if "?" in s), sents[-1] if sents else message.strip())
    return message.strip(), question


def _user(context: str, question: str, spans: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(spans))
    return (f'Message:\n"""\n{context}\n"""\n\nThe user is asking: {question}\n\n'
            f"For each detail below, reply RELEVANT if answering the question depends on this "
            f"specific detail, or NOT if it does not.\n\nDetails:\n{numbered}\n\n"
            f'Reply with exactly one line per detail: "N. RELEVANT" or "N. NOT". No other text.')


def _parse(out: str) -> dict[int, bool]:
    preds, idx = {}, 0
    for line in out.splitlines():
        s = line.strip().lower()
        if s and re.match(r"^\s*(?:\d+|n)?\s*[.):]?\s*(relevant|not)\b", s):
            idx += 1
            preds[idx] = "not" not in s  # the line says RELEVANT (keep) unless it says NOT
    return preds


def judge(message: str, spans: list[str], model: str, timeout: int = 30) -> dict[str, bool]:
    """Return {span_text: keep_bool} for the given spans. keep=True means RELEVANT (do NOT scrub).
    Returns {} on any failure so the caller can fall back to the keyword keep-gate."""
    spans = list(dict.fromkeys(spans))  # de-dup, preserve order
    if not spans:
        return {}
    context, question = _split(message)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": _user(context, question, spans)}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        out = resp.json()["message"]["content"]
    except Exception:
        return {}  # Ollama unavailable / slow -> caller falls back to the deterministic gate
    preds = _parse(out)
    # Missing verdict -> keep (matches the eval's default; parse misses are rare). The judge is
    # utility-leaning by design; residual leaks of kept-incidental spans are the substitution
    # layer's job to generalize.
    return {s: preds.get(i + 1, True) for i, s in enumerate(spans)}
