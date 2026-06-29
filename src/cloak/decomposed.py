"""Orchestrator for the v0.1 decomposed pipeline.

Pipeline:
  Stage 1 — Regex detection (deterministic, from `detectors.py`)
  Stage 2 — LLM span finder (line-format output, parsed deterministically)
  Stage 3 — Deterministic merge + pseudonym assignment (from `pseudonyms.py`)
  Stage 4 — LLM rewriter (prose output with <gen> tags)
  Stage 5 — Deterministic parse + replacement map construction (from `parse.py`)

If Stages 1+2 find no sensitive spans, the rewriter is not invoked. The raw
message passes through unchanged. This makes "no-op" a structural property,
not a behavior we hope the model exhibits.

See `docs/decisions/0003-decomposed-pipeline.md`.
"""

import os
import re

from .detectors import (
    find_called_entities,
    find_codes,
    find_contextual_names,
    find_structured_ids,
    find_workplaces,
    run_detectors,
    run_stage1_detectors,
)
from .models import KEEP_CATEGORIES, Replacement, TransformResult
from .parse import parse_rewriter_output
from .pseudonyms import assign_pseudonyms
from .span_finder import call_span_finder

# NOTE: `rewriter` is an eval-only module that is not bundled with the library. It is
# imported lazily inside `transform_decomposed` (the only consumer, and not part of the
# fast_scrub path), so importing this module works without the eval deps installed.

# Map detector names (from `detectors.py`) to category names used in the pipeline.
DETECTOR_TO_CATEGORY: dict[str, str] = {
    "email": "email",
    "phone": "phone",
    "api_key": "secret",
    "card_number": "account_id",
    "ssn": "account_id",
    "age": "exact_date",
    "ratio": "exact_amount",
}

# Spans that never identify anyone. The span finder prompt lists these in its
# DO-NOT-INCLUDE section, but 7B-class models honor that unreliably — they flag
# `deployments`, `staff engineer`, `A1C`, `MRI` anyway, which then corrupts the
# rewrite. We enforce the exclusions deterministically here rather than hoping
# the model complies. This mirrors the dataset spec's essential-context
# categories; the VALUE / drug / provider / institution sitting next to one of
# these is still flagged separately and generalized.
_NEVER_IDENTIFY: set[str] = {
    # generic medical test / procedure names (the value, drug, provider stay flagged)
    "a1c", "mri", "ct", "ct scan", "x-ray", "xray", "ekg", "ecg", "eeg",
    "ultrasound", "blood test", "hearing test", "cholesterol", "mammogram",
    "biopsy", "scan", "bloodwork", "blood work", "blood pressure",
    # generic activities, topics, life events
    "deployment", "deployments", "weekend", "weekends", "weekday", "on-call",
    "oncall", "on call", "custody", "divorce", "marriage", "wedding", "funeral",
    "layoffs", "layoff", "promotion", "therapy", "kids", "children", "child",
    "moving", "deadline", "graduation",
}

# Role titles / seniority, including compounds (e.g. "staff engineer"). Matched
# against the WHOLE span so a real name like "Dr. Patel" is never dropped.
_ROLE_TITLE_RE = re.compile(
    r"^(a |an |the |my |her |his |their |our )?"
    r"(chief |senior |staff |principal |lead |junior |associate |head |"
    r"deputy |assistant |vice |executive )*"
    r"(engineer|manager|director|analyst|designer|developer|scientist|"
    r"consultant|recruiter|pediatrician|audiologist|nurse|doctor|lawyer|"
    r"attorney|accountant|teacher|professor|architect|officer|administrator|"
    r"coordinator|specialist|intern|president|vp|ceo|cto|cfo|coo)s?$",
    re.IGNORECASE,
)


def _is_non_identifying(span: str) -> bool:
    s = span.strip().lower()
    return s in _NEVER_IDENTIFY or bool(_ROLE_TITLE_RE.match(s))


def _soft_regex_enabled() -> bool:
    """Soft-entity regex heuristics are OFF by default (the model owns soft entities).
    Set CLOAK_SOFT_REGEX=1 to restore them for a weak model that needs recall help."""
    v = os.getenv("CLOAK_SOFT_REGEX")
    return bool(v) and v not in ("0", "false", "False")


def _regex_spans(raw_message: str) -> list[dict]:
    """Run deterministic detectors (hard identifiers + Stage-1 ages/ratios)."""
    spans: list[dict] = []
    detected = {**run_detectors(raw_message), **run_stage1_detectors(raw_message)}
    for detector_name, matches in detected.items():
        category = DETECTOR_TO_CATEGORY.get(detector_name, "other_identifier")
        for match in matches:
            spans.append({"span": match, "category": category})
    for sid in find_structured_ids(raw_message):
        spans.append({"span": sid, "category": "account_id"})
    for code in find_codes(raw_message):
        spans.append({"span": code, "category": "account_id"})
    # Soft-entity regex heuristics (lowercase name/employer guessing) were the dominant
    # false-positive source ("i'm frustrated" -> name). The 2026-06-05 pre-step showed
    # dropping them lifts 7b clean-pass 70%->88% (corruption 26%->9%, over-scrub 35%->11%)
    # with leaks flat. So the architecture is now: the MODEL owns soft entities and the
    # regex owns only the perfect-precision HARD identifiers above. Opt back in with
    # CLOAK_SOFT_REGEX=1 for a weak model that needs the recall crutch (raw 3b).
    if _soft_regex_enabled():
        for ent in find_called_entities(raw_message):
            spans.append({"span": ent, "category": "employer"})
        for ent in find_workplaces(raw_message):
            spans.append({"span": ent, "category": "employer"})
        for nm in find_contextual_names(raw_message):
            spans.append({"span": nm, "category": "name"})
    return spans


def _drop_non_identifying(spans: list[dict]) -> list[dict]:
    """Deterministically drop spans the spec treats as essential context."""
    return [s for s in spans if not _is_non_identifying(s["span"])]


def _split_at(spans: list[dict]) -> list[dict]:
    """Split a fused "<person/thing> at <institution>" span into two spans.

    The span finder sometimes returns e.g. "Dr. Patel at UCLA" as one span; the
    rewriter then generalizes the person but edits the institution OUTSIDE its
    tag (an unmapped change -> diff-coverage failure). Splitting gives each part
    its own span and tag. The institution half becomes `location`.
    """
    out: list[dict] = []
    for s in spans:
        parts = re.split(r"\s+at\s+", s["span"], maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            out.append({"span": parts[0].strip(), "category": s["category"]})
            out.append({"span": parts[1].strip(), "category": "location"})
        else:
            out.append(s)
    return out


# Trailing "unit" words that belong to an amount phrase ("$280k base",
# "30% more pay"). The span finder flags only the number, but the natural
# generalization covers the whole phrase, so the rewriter drops the unit ->
# diff-coverage. We swallow it into the span to keep the map consistent.
_AMOUNT_TAIL_RE = re.compile(
    r"^\s+(base salary|base|salary|raise|more pay|per year|a year|/year|/yr)\b",
    re.IGNORECASE,
)


def _expand_amount_spans(spans: list[dict], raw_message: str) -> list[dict]:
    """Extend an exact_amount span to include a trailing unit word."""
    out: list[dict] = []
    for s in spans:
        if s["category"] == "exact_amount":
            idx = raw_message.find(s["span"])
            if idx != -1:
                m = _AMOUNT_TAIL_RE.match(raw_message[idx + len(s["span"]):])
                if m:
                    s = {"span": s["span"] + m.group(0), "category": s["category"]}
        out.append(s)
    return out


def _dedup_spans(spans: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for s in spans:
        key = (s["span"], s["category"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# Hard identifiers handled DETERMINISTICALLY — never sent to the 7B rewriter,
# which on dense prompts passed a court case number through verbatim (see
# docs/explorations/2026-06-03-attribution-stress-and-combination-gap.md). These
# are distinctive, unique direct identifiers; we substitute a generic placeholder
# in code so a leak cannot depend on rewriter reliability. The rewriter then only
# does the soft natural-language generalizations (locations, employers, dates…).
HARD_CATEGORIES: set[str] = {"email", "phone", "account_id", "secret", "address"}
HARD_PLACEHOLDER: dict[str, str] = {
    "email": "an email address",
    "phone": "a phone number",
    "account_id": "a reference number",
    "secret": "a credential",
    "address": "an address",
}


def _apply_hard(text: str, hard_spans: list[dict]) -> tuple[list[Replacement], str]:
    """Deterministically replace hard identifiers with generic placeholders.

    Returns (replacements, scrubbed_text). Done before the rewriter sees the text,
    so a hard identifier never reaches the LLM and cannot leak through it.
    """
    reps: list[Replacement] = []
    for s in hard_spans:
        span = s["span"]
        if span and span in text:
            ph = HARD_PLACEHOLDER.get(s["category"], "a reference number")
            text = text.replace(span, ph)
            reps.append(Replacement(original=span, replacement=ph, category=s["category"]))
    return reps, text


def _merge_spans(regex_spans: list[dict], llm_spans: list[dict]) -> list[dict]:
    """Combine regex and LLM spans, deduplicating by (span, category)."""
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    # Regex first — they're more reliable on what they cover.
    for span in regex_spans + llm_spans:
        key = (span["span"], span["category"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(span)
    return merged


def transform_decomposed(
    local_model: str,
    span_finder_prompt: str,
    rewriter_prompt: str,
    raw_message: str,
) -> tuple[TransformResult, dict[str, int]]:
    """Run the full decomposed pipeline. Returns (TransformResult, stage_timings_ms)."""
    from .rewriter import build_substitutions_user_message, call_rewriter  # eval-only dep (lazy)

    timings: dict[str, int] = {}

    # Stage 1: deterministic regex detection
    regex_spans = _regex_spans(raw_message)

    # Stage 2: LLM span finder
    try:
        llm_spans, span_raw_output, span_ms = call_span_finder(
            local_model, span_finder_prompt, raw_message
        )
        timings["span_finder_ms"] = span_ms
    except Exception as e:
        return (
            TransformResult(
                protected_message="",
                replacements=[],
                raw_local_output="",
                parse_ok=False,
                parse_error=f"span_finder failed: {e}",
            ),
            timings,
        )

    # Stage 3: merge + deterministic span normalization
    all_spans = _merge_spans(regex_spans, llm_spans)
    all_spans = _split_at(all_spans)               # "Dr. Patel at UCLA" -> two spans
    all_spans = _expand_amount_spans(all_spans, raw_message)  # "$280k" -> "$280k base"
    all_spans = _drop_non_identifying(all_spans)   # drop A1C, staff engineer, deployments…
    # Zone (ADR 0004): keep non-attributable specifics (amounts, dates) verbatim —
    # don't hand them to the rewriter to generalize. Better answers, no privacy
    # cost. Combination-aware exceptions are deferred to Phase 3.
    all_spans = [s for s in all_spans if s["category"] not in KEEP_CATEGORIES]
    all_spans = _dedup_spans(all_spans)

    # No-op short-circuit: nothing identifying at all -> original passes through.
    if not all_spans:
        return (
            TransformResult(
                protected_message=raw_message,
                replacements=[],
                raw_local_output=(
                    f"SPAN_FINDER:\n{span_raw_output}\n\nREWRITER: skipped (no spans detected)"
                ),
                parse_ok=True,
            ),
            timings,
        )

    # Floor: hard identifiers (case/account numbers, emails, phones, secrets,
    # addresses) are scrubbed DETERMINISTICALLY here and never sent to the
    # rewriter. The rewriter then handles only the soft natural-language spans.
    hard_spans = [s for s in all_spans if s["category"] in HARD_CATEGORIES]
    soft_spans = [s for s in all_spans if s["category"] not in HARD_CATEGORIES]
    hard_replacements, working_message = _apply_hard(raw_message, hard_spans)

    name_spans = [s for s in soft_spans if s["category"] == "name"]
    generalize_spans = [s for s in soft_spans if s["category"] != "name"]
    # Number the generalize spans in text order (the rewriter tags by output
    # position; aligning the numbering avoids mislabels). Sort against the
    # working (hard-scrubbed) message — that is what the rewriter receives.
    generalize_spans.sort(key=lambda s: working_message.find(s["span"]))
    name_subs = assign_pseudonyms([s["span"] for s in name_spans], working_message)

    # If only hard identifiers were found, the deterministic scrub is the whole
    # transformation — no rewriter call needed.
    if not soft_spans:
        return (
            TransformResult(
                protected_message=working_message,
                replacements=hard_replacements,
                raw_local_output=(
                    f"SPAN_FINDER:\n{span_raw_output}\n\nHARD SCRUB ONLY (no soft spans)"
                ),
                parse_ok=True,
            ),
            timings,
        )

    # Stage 4: LLM rewriter (on the hard-scrubbed message, soft spans only)
    user_msg, generalize_numbers = build_substitutions_user_message(
        working_message, name_subs, generalize_spans
    )
    try:
        rewriter_output, rewriter_ms = call_rewriter(local_model, rewriter_prompt, user_msg)
        timings["rewriter_ms"] = rewriter_ms
    except Exception as e:
        return (
            TransformResult(
                protected_message="",
                replacements=[],
                raw_local_output=(
                    f"SPAN_FINDER:\n{span_raw_output}\n\nREWRITER FAILED: {e}"
                ),
                parse_ok=False,
                parse_error=f"rewriter failed: {e}",
            ),
            timings,
        )

    # Stage 5: parse + construct map (hard replacements + names + soft generalizations)
    protected_message, soft_replacements = parse_rewriter_output(
        rewriter_output, name_subs, generalize_spans, generalize_numbers
    )
    replacements = hard_replacements + soft_replacements

    combined_raw = (
        f"SPAN_FINDER:\n{span_raw_output}\n\n"
        f"HARD_SCRUB:\n{[(r.original, r.replacement) for r in hard_replacements]}\n\n"
        f"REWRITER_USER_MSG:\n{user_msg}\n\nREWRITER_OUTPUT:\n{rewriter_output}"
    )

    return (
        TransformResult(
            protected_message=protected_message,
            replacements=replacements,
            raw_local_output=combined_raw,
            parse_ok=True,
        ),
        timings,
    )
