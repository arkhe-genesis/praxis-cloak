"""Model test matrix — verify every (provider, model, reasoning-param) combo in the catalog.

WHY: provider model ids and reasoning params drift and confuse (Anthropic effort = a
thinking-token budget; OpenAI effort = a Responses-API enum; names change between
snapshots). This suite is catalog-driven — it enumerates `providers.CATALOG` and each
model's reasoning options — so adding a model/provider in `providers.py` is automatically
covered here. Two layers, sharing `enumerate_cases`:

  * Offline shape tests (no keys, deterministic, CI gate): `backend/tests/test_model_shapes.py`
    stubs the SDK clients and asserts the exact request kwargs per combo. Catches *our*
    regressions (renamed param, wrong API surface, bad budget math).
  * Live smoke (this module's default): actually calls each combo with a tiny prompt and
    checks a non-empty reply streams back. Catches *provider-side* issues (model id gone,
    param rejected, auth). Skips providers whose key isn't loaded.

Run live:   python -m backend.model_check            (full matrix)
            python -m backend.model_check --quick     (one call per model: its default)
            python -m backend.model_check --json      (machine-readable)
            python -m backend.model_check --doc       (regenerate docs/reference/cloud-models.md)
The convenience wrapper `scripts/model-check.sh` sources your keys first.
Exit code is non-zero if any non-skipped case fails (so it can gate CI).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import providers
from .config import Config, load_config
from .transport import DirectTransport, OpenAITransport

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOC = _REPO_ROOT / "docs" / "reference" / "cloud-models.md"
_DEFAULT_PROMPT = "Reply with the single word: ok"


@dataclass
class Case:
    provider: str
    model: str
    model_label: str
    param: str  # the reasoning control's name ("Effort"/"Intelligence"/"Thinking") or ""
    effort: str  # provider value ("" if the model has no reasoning control)
    effort_label: str  # the user-facing label ("" if none)
    available: bool  # the provider's key is loaded


def enumerate_cases(cfg: Config, *, quick: bool = False) -> list[Case]:
    """The full test matrix, derived entirely from the catalog. `quick` keeps only each
    model's default reasoning option (one call per model)."""
    cases: list[Case] = []
    for m in providers.CATALOG:
        avail = providers.provider_available(m.provider, cfg)
        if m.reasoning is None:
            cases.append(Case(m.provider, m.id, m.label, "", "", "", avail))
            continue
        opts = m.reasoning.options
        if quick:
            opts = tuple(o for o in opts if o.value == m.reasoning.default) or opts[:1]
        for o in opts:
            cases.append(
                Case(m.provider, m.id, m.label, m.reasoning.label, o.value, o.label, avail)
            )
    return cases


@dataclass
class Result:
    case: Case
    status: str  # "ok" | "fail" | "skip"
    ms: int
    sample: str
    error: str


def run_case(case: Case, cfg: Config, *, prompt: str, max_chars: int) -> Result:
    """Live: build the transport for this combo and stream a tiny reply. Non-empty reply =
    the request shape is accepted by the provider."""
    if not case.available:
        return Result(case, "skip", 0, "", "provider key not loaded")
    transport = providers.build_transport(case.model, case.effort or None, cfg)
    messages = [{"role": "user", "content": prompt}]
    started = time.monotonic()
    try:
        acc = ""
        for chunk in transport.stream(messages):
            acc += chunk
            if len(acc) >= max_chars:
                break
        ms = int((time.monotonic() - started) * 1000)
        text = acc.strip()
        if not text:
            return Result(case, "fail", ms, "", "empty reply (no text deltas)")
        return Result(case, "ok", ms, _oneline(text, max_chars), "")
    except Exception as e:  # noqa: BLE001 — we want to capture any provider/SDK error verbatim
        ms = int((time.monotonic() - started) * 1000)
        return Result(case, "fail", ms, "", f"{type(e).__name__}: {e}")


def _oneline(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def collect(cfg: Config, *, quick: bool, prompt: str, max_chars: int,
            provider: str | None, model: str | None) -> list[Result]:
    cases = enumerate_cases(cfg, quick=quick)
    if provider:
        cases = [c for c in cases if c.provider == provider]
    if model:
        cases = [c for c in cases if model.lower() in c.model.lower()]
    return [run_case(c, cfg, prompt=prompt, max_chars=max_chars) for c in cases]


# -- output ---------------------------------------------------------------------

_MARK = {"ok": "✓", "fail": "✗", "skip": "·"}


def format_table(results: list[Result]) -> str:
    rows = []
    for r in results:
        c = r.case
        val = c.effort_label or "—"
        last = r.error if r.status != "ok" else r.sample
        rows.append(
            (_MARK[r.status], c.provider, c.model_label, c.param or "—", val,
             f"{r.ms}ms" if r.ms else "", last)
        )
    headers = ("", "provider", "model", "param", "value", "time", "result")
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [0] * 7
    out = []
    out.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    out.append("  ".join("-" * w for w in widths))
    for row in rows:
        # don't pad the trailing free-text column
        out.append("  ".join(str(x).ljust(w) for x, w in zip(row[:-1], widths[:-1])) + "  " + str(row[-1]))
    ok = sum(r.status == "ok" for r in results)
    fail = sum(r.status == "fail" for r in results)
    skip = sum(r.status == "skip" for r in results)
    out.append("")
    out.append(f"{ok} ok · {fail} fail · {skip} skipped (no key)  of {len(results)} cases")
    return "\n".join(out)


def format_json(results: list[Result]) -> str:
    return json.dumps([{**asdict(r.case), "status": r.status, "ms": r.ms,
                        "sample": r.sample, "error": r.error} for r in results], indent=2)


# -- canonical doc --------------------------------------------------------------

def render_doc(cfg: Config) -> str:
    """Generate the canonical model/param reference from the catalog + transport constants,
    so it can't drift from the code. Live verification is stamped by running this suite."""
    adaptive = [m for m in providers.CATALOG if m.thinking_mode == "adaptive"]
    budget_models = [m for m in providers.CATALOG if m.thinking_mode == "budget"]
    lines: list[str] = []
    lines.append("# Cloud models & reasoning params (canonical)")
    lines.append("")
    lines.append("> Generated from `backend/providers.py` by `python -m backend.model_check --doc`.")
    lines.append("> Do not hand-edit the tables — change the catalog and regenerate. Verify the")
    lines.append("> live shapes with `python -m backend.model_check` (or `scripts/model-check.sh`).")
    lines.append("")
    lines.append("The app talks to one cloud model at a time, picked in the composer. A model belongs")
    lines.append("to a **provider** (available iff its key env is set) and usually exposes one")
    lines.append("**reasoning control**. Selection flows as `(model, effort)` per `/send`; the")
    lines.append("transport (`backend/transport.py`) turns `effort` into the provider's native param.")
    lines.append("")
    for p in providers.PROVIDERS:
        models = [m for m in providers.CATALOG if m.provider == p.id]
        if not models:
            continue
        lines.append(f"## {p.label} (`{p.id}`) — key env `{p.key_env}`")
        lines.append("")
        lines.append("| Model id | Label | Control | Values (label → api) | Default |")
        lines.append("| --- | --- | --- | --- | --- |")
        for m in models:
            if m.reasoning:
                vals = ", ".join(f"{o.label}→`{o.value}`" for o in m.reasoning.options)
                ctrl = m.reasoning.label
                default = f"`{m.reasoning.default}`"
            else:
                vals, ctrl, default = "—", "—", "—"
            lines.append(f"| `{m.id}` | {m.label} | {ctrl} | {vals} | {default} |")
        lines.append("")
    lines.append("## Request shapes (how `effort` is applied)")
    lines.append("")
    lines.append("Anthropic reasoning differs by model generation (the SDK's `text_stream` yields")
    lines.append("only the visible answer either way, so the rest of the pipeline is unchanged):")
    lines.append("")
    adaptive_ids = ", ".join(f"`{m.id}`" for m in adaptive) or "—"
    budget_ids = ", ".join(f"`{m.id}`" for m in budget_models) or "—"
    lines.append(f"**Adaptive** ({adaptive_ids}) — newer API; the effort value IS the API enum,")
    lines.append("sent via `extra_body`:")
    lines.append("")
    lines.append("```python")
    lines.append("client.messages.stream(")
    lines.append(f"    model=<id>, messages=<messages>, max_tokens={DirectTransport._ADAPTIVE_MAX_TOKENS},")
    lines.append('    extra_body={"thinking": {"type": "adaptive"},')
    lines.append('               "output_config": {"effort": <value>}},  # low|medium|high|xhigh|max')
    lines.append(")")
    lines.append("```")
    lines.append("")
    budget_tbl = ", ".join(f"`{k}`→{v}" for k, v in DirectTransport._BUDGET.items())
    lines.append(f"**Budget** ({budget_ids}) — older API; a Standard/Extended toggle. "
                 f"`extended` → `thinking={{type:enabled, budget_tokens}}` ({budget_tbl}),")
    lines.append("`max_tokens = budget + max(base, 4096)`; `standard` sends **no** `thinking` block.")
    lines.append("")
    lines.append("**OpenAI** — reasoning models use the **Responses API**:")
    lines.append("")
    lines.append("```python")
    lines.append("client.responses.create(")
    lines.append("    model=<id>, input=<messages>, stream=True,")
    lines.append('    reasoning={"effort": <value>},   # none | minimal | low | medium | high | xhigh')
    lines.append(f"    max_output_tokens={OpenAITransport._REASONING_MAX_OUTPUT},  # reserve room for reasoning tokens")
    lines.append(")")
    lines.append("# stream: yield on events of type 'response.output_text.delta'")
    lines.append("```")
    lines.append("")
    lines.append("## Adding a provider or model")
    lines.append("")
    lines.append("1. (New provider) add a `ChatTransport` subclass in `backend/transport.py` and a")
    lines.append("   `ProviderSpec` + key env in `backend/providers.py`; route it in `build_transport`.")
    lines.append("2. Add `ModelSpec`(s) to `CATALOG` with a `Reasoning` control if applicable.")
    lines.append("3. Add a per-provider shape expectation in `backend/tests/test_model_shapes.py`")
    lines.append("   (the test fails loudly for any provider without one).")
    lines.append("4. Run `python -m backend.model_check` to verify live, then `--doc` to regenerate")
    lines.append("   this file.")
    lines.append("")
    return "\n".join(lines)


def write_doc(cfg: Config, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_doc(cfg))
    return path


# -- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the cloud model/param matrix.")
    ap.add_argument("--quick", action="store_true", help="one call per model (its default effort)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--doc", action="store_true", help="regenerate the canonical doc and exit")
    ap.add_argument("--doc-path", default=str(_DEFAULT_DOC))
    ap.add_argument("--provider", help="only this provider id")
    ap.add_argument("--model", help="only models whose id contains this substring")
    ap.add_argument("--prompt", default=_DEFAULT_PROMPT)
    ap.add_argument("--max-chars", type=int, default=120)
    args = ap.parse_args(argv)

    cfg = load_config()

    if args.doc:
        path = write_doc(cfg, Path(args.doc_path))
        print(f"wrote {path}")
        return 0

    results = collect(
        cfg, quick=args.quick, prompt=args.prompt, max_chars=args.max_chars,
        provider=args.provider, model=args.model,
    )
    print(format_json(results) if args.json else format_table(results))
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
