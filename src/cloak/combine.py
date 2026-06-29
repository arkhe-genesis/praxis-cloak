"""Combination breaker: the minimum-necessary search using the local proxies,
end-to-end — detect candidates, decide the minimum set, render grammatically.

The breaker is the *combination* layer that sits on top of the base direct-identifier
scrub. Direct identifiers (names -> pseudonyms, emails/phones/accounts -> placeholders)
are handled deterministically first; the breaker then decides which of the remaining
*quasi-identifiers* to generalize so the residual profile is no longer attributable —
while changing as little as possible.

Three on-device signals gate every generalization:

- **task-function** ([`taskfn`](taskfn.py)): only GENERAL details are candidates. A
  detail the answer needs (SPECIFIC) is never stripped.
- **local re-id** ([`reid_local`](reid_local.py)): generalize a candidate ONLY IF doing
  so *reduces* re-id risk. This uses the 7B's reliable DIRECTIONAL signal and ignores
  its unreliable absolute calibration, so an over-conservative proxy can't over-scrub.
- **faithfulness backstop** ([`faithfulness`](faithfulness.py)): never generalize a
  candidate if doing so drops one of the user's core points. When the identifier *is*
  the point (the un-breakable case), it is HELD and surfaced for the preview rather than
  silently gutted.

End-to-end entry point: `run_breaker`. The decision core is `break_combination`;
candidate discovery is `detect_candidates`; grammatical rendering is `render_protected`
(reuses the same rewriter the base pipeline uses — no naive string substitution).

See docs/explorations/2026-06-03-task-function-ablation.md and
docs/explorations/2026-06-03-faithfulness-guardrail.md.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from .decomposed import (
    HARD_CATEGORIES,
    HARD_PLACEHOLDER,
    _dedup_spans,
    _drop_non_identifying,
    _expand_amount_spans,
    _merge_spans,
    _regex_spans,
    _split_at,
)
from .models import KEEP_CATEGORIES, Replacement
from .parse import parse_rewriter_output
from .pseudonyms import assign_pseudonyms
from . import relevance, synthetics
from .detectors import API_KEY, CARD, EMAIL, PHONE, SSN
from .span_finder import call_span_finder

# NOTE: faithfulness, reid_local, rewriter, and taskfn are EVAL-ONLY modules that are
# not bundled with the library. They are imported lazily inside the breaker functions
# (run_breaker / break_combination / detect_candidates / render_protected /
# _classify_residual) so that `import cloak` and the fast_scrub path work without
# them. Calling those breaker functions without the eval deps installed raises a clear
# ImportError at the point of use, which is the intended behavior.


def _risk_rank() -> dict:
    """Lazy {risk: index} ranking map (needs the eval-only reid_local.RISK_ORDER)."""
    from .reid_local import RISK_ORDER

    return {r: i for i, r in enumerate(RISK_ORDER)}

# Generalize the most-identifying categories first.
_PRIORITY = {
    "employer": 1, "location": 2, "name": 3,
    "medical_specifics": 4, "legal_specifics": 4, "other_identifier": 5,
}

# Generic stand-ins used during the re-id reduction TEST and faithfulness TEST.
# They only need to be a plausible generalization so the directional re-id signal
# and the "did this drop the core?" signal are right — the final, grammatical
# generalization is produced by the rewriter (see render_protected).
_GENERIC = {
    "employer": "a company",
    "location": "a city",
    "medical_specifics": "a health condition",
    "legal_specifics": "a legal matter",
    "other_identifier": "someone",
}


def _generic_for(category: str) -> str:
    return _GENERIC.get(category, "a person")


# Soft named-entity categories that are always generalized when detected (the
# "floor"), independent of the noisy re-id score — unless task-function marks them
# load-bearing (SPECIFIC) or the faithfulness backstop holds them as core.
_FLOOR_CATEGORIES = {"employer", "location"}

# Common words that are also first names — never globally rewrite a name COMPONENT
# that matches one of these ("Will Smith" must not turn every "will" into a pseudonym).
# The full name is still replaced as a whole; only bare-component replacement is guarded.
_COMMON_WORDS = {
    "will", "may", "mark", "rose", "grace", "hope", "faith", "joy", "april", "june",
    "july", "august", "dawn", "summer", "autumn", "sunny", "drew", "chase", "hunter",
    "miles", "ray", "art", "bill", "frank", "jack", "guy", "max", "gene", "rich",
    "mercy", "honor", "reign", "royal", "angel", "amber", "ivy", "olive", "crystal",
}


# Role / party / relation nouns the span-finder sometimes mislabels as `name`
# ("plaintiff", "my manager"). Pseudonymizing one of these invents a person
# ("my manager" -> "my Liam"). We exclude these by NAME, not by capitalization:
# real user input is often lowercase ("i'm charlie callinan"), and failing to
# pseudonymize a real name is the worst failure — so we bias toward pseudonymizing
# anything the finder flagged as a name unless it's a known non-name word.
_NOT_A_NAME = {
    # legal/party roles
    "plaintiff", "defendant", "claimant", "respondent", "petitioner", "appellant",
    "complainant", "witness", "victim", "patient", "client", "customer", "user",
    # relations / generic people
    "mom", "mum", "dad", "mother", "father", "parent", "parents", "son", "daughter",
    "kid", "kids", "child", "children", "brother", "sister", "sibling", "wife",
    "husband", "spouse", "partner", "boyfriend", "girlfriend", "friend", "colleague",
    "boss", "manager", "coworker", "neighbor", "neighbour", "team", "everyone",
}
# Internet slang / interjections the span finder sometimes mislabels as a NAME on short,
# casual turns ("vibin" -> faked to "Alex" corrupted the message). A deterministic drop here,
# same spirit as _NOT_NAME_TOKEN — fixes it whatever the detector does, and fuzz_detect.py guards
# the regression. These are never real first names. (v7 training fixed most, e.g. bingus/bruh/ngl;
# this catches the residual the model still over-fires on.)
_NOT_NAME_SLANG = {
    "vibin", "vibing", "vibes", "bruh", "bruv", "ngl", "lowkey", "highkey", "deadass",
    "yeet", "bingus", "fr", "frfr", "tbh", "smh", "imo", "ikr", "istg", "lmao", "lmfao",
    "rofl", "lolol", "yolo", "fomo", "sus", "slay", "based", "cringe", "mid", "goated",
    "rizz", "delulu", "yass", "yasss", "oof", "meh", "ugh", "innit", "sheesh", "yapping",
}


def _looks_like_name(span: str) -> bool:
    """True unless the span is a known role/relation word or slang, not a real name.

    Deliberately does NOT require capitalization — lowercase real names ("charlie
    callinan") must still pseudonymize. Failing to pseudonymize a real name is the
    catastrophic failure, so we bias toward treating finder-flagged names as names.
    """
    s = span.strip().lower()
    if not s:
        return False
    return not all(w in _NOT_A_NAME or w in _NOT_NAME_SLANG for w in s.split())


# Tokens that are never part of a personal name — verbs, emotion/state adjectives,
# function words, and event nouns that the span detectors sometimes swallow into a
# name span ("brother mike OWES" -> two tokens; "i'm FRUSTRATED" -> a false name).
# Trimming these at the substitution boundary fixes the corruption uniformly, whatever
# detector produced the span, instead of growing each detector's blocklist forever.
# Homonym names (_COMMON_WORDS: will/drew/grace/may…) are EXCLUDED below so a real
# "drew" still pseudonymizes — that residual (a name that is also a verb) is the ~10%
# this cannot resolve without a part-of-speech tagger.
_NOT_NAME_VERBS = {
    "owe", "owes", "owed", "owing", "borrow", "borrows", "borrowed", "lend", "lends", "lent",
    "pay", "pays", "paid", "paying", "invite", "invites", "invited", "inviting",
    "call", "calls", "called", "calling", "text", "texts", "texted", "texting",
    "email", "emails", "emailed", "message", "messages", "messaged", "ask", "asks", "asked",
    "tell", "tells", "told", "telling", "want", "wants", "wanted", "need", "needs", "needed",
    "keep", "keeps", "kept", "keeping", "live", "lives", "lived", "living",
    "work", "works", "worked", "working", "say", "says", "said", "saying",
    "go", "goes", "went", "going", "come", "comes", "came", "coming",
    "make", "makes", "made", "making", "get", "gets", "got", "getting",
    "give", "gives", "gave", "giving", "take", "takes", "took", "taking",
    "think", "thinks", "thought", "feel", "feels", "felt", "feeling",
    "seem", "seems", "seemed", "move", "moves", "moved", "moving",
    "start", "starts", "started", "starting", "help", "helps", "helped",
    "see", "sees", "saw", "seeing", "know", "knows", "knew", "knowing",
    "leave", "leaves", "left", "leaving", "send", "sends", "sent", "sending",
    "meet", "meets", "met", "snapped", "snaps", "questioned", "questions",
    "copied", "copies", "copying", "ghosted", "ghosting", "cancel", "cancels",
    "canceled", "cancelled", "canceling", "cancelling", "threw", "throws", "throwing",
    "brought", "brings", "refuses", "refused", "refuse", "stays", "stayed", "stay",
    "broke", "breaks", "earns", "earned", "earn", "spent", "spends", "spend",
    "charges", "charged", "charge", "promised", "promises", "promise",
    "mentioned", "mentions", "mention", "decided", "decides", "decide", "lives",
}
_NOT_NAME_STATES = {
    "frustrated", "exhausted", "annoyed", "overwhelmed", "anxious", "worried", "upset",
    "angry", "sad", "happy", "nervous", "scared", "afraid", "confused", "stressed",
    "tired", "hurt", "devastated", "furious", "guilty", "ashamed", "embarrassed",
    "disappointed", "hopeful", "excited", "glad", "relieved", "conflicted", "resentful",
    "defensive", "hesitant", "unsure", "fine", "okay", "sorry", "ready", "done", "busy",
    "stuck", "lost", "torn", "drained", "main", "sole", "only", "new", "old", "close",
    "former", "current", "late", "early",
}
_NOT_NAME_FUNC = {
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "it", "its", "we", "us", "our", "the", "a", "an", "and", "or", "but", "so",
    "to", "at", "in", "on", "of", "for", "with", "from", "this", "that", "these", "those",
    "is", "was", "are", "am", "be", "been", "who", "here", "there", "just", "really",
    "also", "too", "now", "then", "when", "how", "what", "why", "where", "not", "no",
    "im", "dont", "about", "because", "since", "always", "never", "still", "very", "quite",
}
_NOT_NAME_NOUNS = {
    "birthday", "wedding", "party", "funeral", "graduation", "anniversary", "reunion",
    "meeting", "money", "rent", "loan", "favor", "contact", "stuff", "plans", "credit",
}
_NOT_NAME_TOKEN = (_NOT_NAME_VERBS | _NOT_NAME_STATES | _NOT_NAME_FUNC | _NOT_NAME_NOUNS) - _COMMON_WORDS


def _name_core(span: str) -> str:
    """Trim leading/trailing non-name tokens from a detected name span, leaving the
    actual name. "mike owes" -> "mike"; "frustrated" -> "" (a false positive, dropped
    by the caller). Homonym names survive (excluded from the lexicon above)."""
    toks = span.split()

    def is_name(t: str) -> bool:
        w = t.strip(".,!?;:'\"’").lower()
        return bool(w) and w not in _NOT_NAME_TOKEN

    while toks and not is_name(toks[-1]):
        toks.pop()
    while toks and not is_name(toks[0]):
        toks.pop(0)
    return " ".join(toks)


# Hard-identifier shapes scrubbed as a FAIL-CLOSED backstop on the final output,
# regardless of what the detectors found upstream. A missed email/phone/card/SSN is a
# privacy breach, and these shapes are regex-deterministic, so the guarantee must not
# rest on the LLM span finder.
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# A social/messaging handle (@name), but NOT an email's "@domain" (word char before @).
_HANDLE_RE = re.compile(r"(?<![\w@.])@[A-Za-z][A-Za-z0-9_.]{1,}")
# Dates of birth / explicit dates: MM/DD/YYYY, M/D/YY, and ISO YYYY-MM-DD.
_DOB_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_HARD_SWEEP = [
    ("email", EMAIL), ("phone", PHONE), ("account_id", CARD),
    ("account_id", SSN), ("secret", API_KEY), ("phone", _IP_RE),
    ("handle", _HANDLE_RE), ("dob", _DOB_RE),
]


def _is_codey(span: str) -> bool:
    """A real code/account/secret has a digit, an uppercase letter, or a separator.
    Plain lowercase words ("drowning", "too sensitive") that the LLM mislabels as an
    identifier do not — so we never fake them into gibberish."""
    return any(c.isdigit() or c.isupper() or c in "-_/" for c in span)


def _sweep_hard_ids(text: str, used: set, mapping: dict) -> tuple[str, list[Replacement]]:
    """Re-scan protected text for hard-identifier shapes; scrub any survivor with a
    realistic fake. Skips values already in `used` (fakes we just inserted)."""
    reps: list[Replacement] = []
    for cat, rx in _HARD_SWEEP:
        for real in {m.group(0) for m in rx.finditer(text)}:
            if real in used:  # a fake we inserted this turn — don't re-scrub it
                continue
            fake = synthetics.fake_for(cat, real, used)
            used.add(fake)
            text = text.replace(real, fake)
            mapping[real] = (fake, cat)
            reps.append(Replacement(original=real, replacement=fake, category=cat))
    return text, reps


# Ages stated in context — a bare number a model reliably misses but regex catches.
# Context-gated to avoid faking non-age numbers (amounts, durations, times).
_AGE_RES = [
    re.compile(r"\b(?:i'?m|im|i\s*am)\s+(\d{2})\b"
               r"(?!\s*(?:min|hour|hr|day|week|month|year(?!s?\s*old)|dollar|percent|"
               r"mile|pound|kg|lb|am|pm|k\b|%|/))", re.IGNORECASE),
    re.compile(r"\bage[d]?\s*:?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|y\.?o\.?\b|yrs?\s*old)\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+(?:son|daughter|kid|child|baby)\s+(?:is|just turned))\s+(\d{1,2})\b",
               re.IGNORECASE),
]


def _sweep_ages(text: str, used: set, mapping: dict) -> tuple[str, list[Replacement]]:
    """Replace ages stated in context with a different plausible age (number only)."""
    reps: list[Replacement] = []
    for rx in _AGE_RES:
        def repl(m: re.Match) -> str:
            num = m.group(1)
            if num in mapping:
                fake = mapping[num][0]
            else:
                fake = synthetics.fake_for("age", num, used)
                used.add(fake)
                mapping[num] = (fake, "age")
            reps.append(Replacement(original=num, replacement=fake, category="age"))
            return m.group(0).replace(num, fake, 1)
        text = rx.sub(repl, text)
    return text, reps


_PAREN_RE = re.compile(r"\(([^)]+)\)")


def _anchor_pin(pin: str, message: str, entity_spans: list[str]) -> str | None:
    """Anchor a re-id pin to a verbatim span of the message, or return None.

    The 7B re-id proxy formats pins several ways: a bare phrase ("the only woman
    on the team"), or a description with the real value in parentheses ("specific
    treatment location (Mayo Clinic in Rochester)"). We try, in order: the
    parenthetical value, the verbatim pin, and the pin with known entities trimmed
    out — taking the first that locates. When an anchor embeds an entity we already
    handle, we keep only the distinctive remainder so the two don't overlap.
    """
    attempts: list[str] = []
    m = _PAREN_RE.search(pin)
    if m:
        attempts.append(m.group(1).strip())
    attempts.append(pin)
    for text in attempts:
        anchor = _locate_pin(text, message)
        if anchor is None:
            continue
        if any(e in anchor for e in entity_spans):
            trimmed = _trim_known_entities(anchor, entity_spans)
            relocated = _locate_pin(trimmed, message) if trimmed else None
            return relocated if relocated else anchor
        return anchor
    trimmed = _trim_known_entities(pin, entity_spans)
    return _locate_pin(trimmed, message) if trimmed else None


# ---------------------------------------------------------------------------
# Item 2 — candidate detection from the message
# ---------------------------------------------------------------------------


def _normalize_spans(message: str, span_finder_model: str, span_prompt: str) -> list[dict]:
    """Run the same Stage-1+2 detection and normalization the base pipeline uses."""
    regex_spans = _regex_spans(message)
    llm_spans, _, _ = call_span_finder(span_finder_model, span_prompt, message)
    spans = _merge_spans(regex_spans, llm_spans)
    spans = _split_at(spans)                       # "Dr. Patel at UCLA" -> two spans
    spans = _expand_amount_spans(spans, message)   # "$280k" -> "$280k base"
    spans = _drop_non_identifying(spans)           # drop A1C, staff engineer, deployments…
    spans = [s for s in spans if s["category"] not in KEEP_CATEGORIES]  # zone
    return _dedup_spans(spans)


def _locate_pin(pin: str, message: str) -> str | None:
    """Locate a re-id pin as a verbatim (case-insensitive) substring of the message.

    Pins are phrases the entity span-finder misses ("the only woman on the team").
    We can only auto-generalize a phrase we can anchor to actual text; a pin we
    cannot locate is returned to the caller to surface, never silently dropped.
    """
    p = pin.strip().strip('"').strip()
    if not p:
        return None
    low = message.lower()
    i = low.find(p.lower())
    if i == -1:
        return None
    return message[i : i + len(p)]


def _trim_known_entities(phrase: str, entity_spans: list[str]) -> str:
    """Remove already-handled entity spans (e.g. 'Meta') from a pin, leaving the
    distinctive remainder ('the only woman on the infrastructure team').

    A combination pin usually embeds an entity we already generalize separately;
    if we kept the whole pin as a candidate it would overlap that entity and the
    sequential search would lose one to the other. Trimming yields a non-overlapping
    combination candidate that survives the entity's generalization.
    """
    out = phrase
    for e in sorted(entity_spans, key=len, reverse=True):
        if e and e in out:
            out = out.replace(e, " ")
    out = re.sub(r"\s+", " ", out).strip(" ,.;:")
    out = re.sub(r"\s+(at|in|for|with|of|on|the|a|an)$", "", out, flags=re.IGNORECASE).strip()
    out = re.sub(r"^(at|in|for|with|of|on|the|a|an)\s+", "", out, flags=re.IGNORECASE).strip()
    return out


# Marked-uniqueness clauses ("the only woman on the team", "the only person in my
# program with a law degree"). The re-id proxy names these but often as a paraphrase
# that won't anchor; this catches them deterministically from the text. Requires a
# nominal context (an article before only/sole/first, or "only one/person/…") so
# "I only want…" doesn't match. Stops at clause punctuation.
_UNIQUENESS_RE = re.compile(
    r"((?:the|a|an)\s+(?:only|sole|first|youngest|oldest)\s+[^.!?;,\n—]*"
    r"|only\s+(?:one|person|woman|man|guy|girl|female|male)\b[^.!?;,\n—]*)",
    re.IGNORECASE,
)


def _uniqueness_candidates(message: str, taken: list[str]) -> list[dict]:
    """Find marked-uniqueness clauses and turn them into generalization candidates.

    The distinctive clause becomes a candidate; task-function / faithfulness / re-id
    then decide whether to generalize it (incidental uniqueness) or keep + surface it
    (the uniqueness IS the point). Entities already handled separately are trimmed out.
    """
    out: list[dict] = []
    for m in _UNIQUENESS_RE.finditer(message):
        clause = m.group(1).strip(" ,.;:")
        if len(clause.split()) < 3:
            continue
        trimmed = _trim_known_entities(clause, taken)
        anchor = _locate_pin(trimmed, message) or _locate_pin(clause, message)
        if anchor is None or len(anchor.split()) < 3:
            continue
        if any(anchor == t or anchor in t for t in taken):
            continue
        taken.append(anchor)
        out.append({"span": anchor, "category": "other_identifier", "generic": "someone"})
    return out


def detect_candidates(
    message: str,
    span_finder_model: str,
    span_prompt: str,
    reid_model: str,
    reid_prompt: str,
) -> dict:
    """Derive the breaker's working set from a message.

    Returns a dict:
      - name_spans:  [{span, category}]  detected names (-> pseudonyms, never broken)
      - hard_spans:  [{span, category}]  direct identifiers (-> deterministic placeholders)
      - candidates:  [{span, category, generic}]  the quasi-identifiers the breaker may
                     generalize: soft, non-name, non-KEEP spans + located re-id pins.
      - unlocated_pins: re-id pins we could not anchor to verbatim text (surfaced).
      - reid:        the raw re-id verdict on the message {risk, pins}.
    """
    from .reid_local import assess_local  # eval-only dep (lazy)

    # Span-finding and the re-id assessment are independent — run them concurrently
    # (span-finding is the slower of the two, so re-id is effectively free here).
    with ThreadPoolExecutor(max_workers=2) as ex:
        reid_fut = ex.submit(lambda: assess_local(reid_model, reid_prompt, message)[0])
        spans = _normalize_spans(message, span_finder_model, span_prompt)
        reid = reid_fut.result()

    # Role/party words mislabeled as names are neither pseudonymized nor generalized.
    name_spans = [s for s in spans if s["category"] == "name" and _looks_like_name(s["span"])]
    hard_spans = [s for s in spans if s["category"] in HARD_CATEGORIES]

    candidates: list[dict] = []
    taken: list[str] = []
    for s in spans:
        if s["category"] == "name" or s["category"] in HARD_CATEGORIES:
            continue
        if s["span"] in taken:
            continue
        taken.append(s["span"])
        candidates.append(
            {"span": s["span"], "category": s["category"], "generic": _generic_for(s["category"])}
        )

    # Augment with re-id combination pins the entity span-finder misses (e.g.
    # "the only woman on the infrastructure team"). For each pin, strip out any
    # entity we already handle separately, then anchor the distinctive remainder
    # to verbatim text. A pin we cannot anchor is surfaced, never silently dropped.
    entity_spans = list(taken)
    unlocated: list[str] = []
    for pin in reid.get("pins", []):
        anchor = _anchor_pin(pin, message, entity_spans)
        if anchor is None:
            unlocated.append(pin)
            continue
        # Dedup: skip only if equal to / contained in an existing candidate (the
        # existing one already covers more). A pin that CONTAINS a smaller entity
        # is a legitimately distinct, larger combination candidate.
        if any(anchor == t or anchor in t for t in taken):
            continue
        taken.append(anchor)
        candidates.append(
            {"span": anchor, "category": "other_identifier", "generic": _generic_for("other_identifier")}
        )

    # Marked-uniqueness clauses the proxy phrased un-anchorably (the most-identifying
    # class). Deterministic from the text, so they become candidates rather than
    # surfaced-but-untouched pins.
    candidates.extend(_uniqueness_candidates(message, taken))

    return {
        "name_spans": name_spans,
        "hard_spans": hard_spans,
        "candidates": candidates,
        "unlocated_pins": unlocated,
        "reid": reid,
    }


def _base_scrub(message: str, name_spans: list[dict], hard_spans: list[dict]) -> tuple[str, dict, list[Replacement]]:
    """Handle direct identifiers deterministically: names -> pseudonyms, hard -> placeholders.

    Returns (base_message, name_subs, base_replacements). The breaker's combination
    search runs on base_message so re-id reflects the *residual* (post-direct-id)
    profile — what the combination actually adds.
    """
    name_subs = assign_pseudonyms([s["span"] for s in name_spans], message)
    base = message
    reps: list[Replacement] = []
    for orig, pseudo in name_subs.items():
        if orig in base:
            base = base.replace(orig, pseudo)
            reps.append(Replacement(original=orig, replacement=pseudo, category="name"))
    for s in hard_spans:
        if s["span"] in base:
            ph = HARD_PLACEHOLDER.get(s["category"], "a reference number")
            base = base.replace(s["span"], ph)
            reps.append(Replacement(original=s["span"], replacement=ph, category=s["category"]))
    return base, name_subs, reps


# ---------------------------------------------------------------------------
# Item 1 — decision core with the faithfulness backstop
# ---------------------------------------------------------------------------


def break_combination(
    message: str,
    candidates: list[dict],
    core: dict,
    task_model: str,
    reid_model: str,
    faith_model: str,
    task_prompt: str,
    reid_prompt: str,
    faith_check_prompt: str,
    target_risk: str = "low",
) -> dict:
    """Run the minimum-necessary combination-breaking search on `message`.

    `message` should already have direct identifiers handled (see _base_scrub).
    `core` is the user's extracted core idea (from the ORIGINAL message), used by
    the faithfulness backstop. `target_risk` is the residual re-id risk we aim for.

    Two passes, both gated by the SAME two safety checks (task-function NEED and the
    faithfulness backstop), so neither can strip a load-bearing or core detail:

    1. **Reduction pass** — greedily generalize a candidate when doing so *measurably*
       reduces re-id risk. This is the minimum-necessary core: change only what helps.
    2. **Close-the-gap pass** — if risk is still above `target_risk` after pass 1,
       generalize the remaining GENERAL + faithfulness-safe candidates even without a
       measured per-step drop. A high/unique combination often needs several details
       removed before the discrete level moves, so a strict per-step rule stalls. This
       is safe by construction: these candidates are certified not-needed (task-fn) and
       not-the-point (faithfulness), so generalizing them cannot hurt the answer or drop
       the user's idea. The pass only fires above target, so low-risk inputs (and the
       controls) are never touched.
    """
    from .faithfulness import check_faithful  # eval-only dep (lazy)
    from .reid_local import assess_local  # eval-only dep (lazy)
    from .taskfn import call_task_function  # eval-only dep (lazy)

    _RANK = _risk_rank()

    # Assess each span's NEED INDEPENDENTLY (one call each). Listing spans together
    # makes the 7B's value/type call order-sensitive (cross-span interference);
    # per-span removes that. POINT is intentionally unused — NEED subsumes it, and
    # the faithfulness backstop (not POINT) now guards the user's core.
    tf: dict[str, dict] = {}
    if candidates:
        def _need(c: dict) -> tuple[str, dict]:
            res, _, _ = call_task_function(task_model, task_prompt, message, [c["span"]])
            return c["span"], res[c["span"]]
        with ThreadPoolExecutor(max_workers=min(len(candidates), 6)) as ex:
            for span, verdict in ex.map(_need, candidates):
                tf[span] = verdict

    gen = [c for c in candidates if tf[c["span"]]["need"] == "general"]
    gen.sort(key=lambda c: _PRIORITY.get(c["category"], 9))

    working = message
    # The search needs only the risk LEVEL after each step -> brief re-id (no pins).
    cur = assess_local(reid_model, reid_prompt, working, brief=True)[0]["risk"]
    applied: list[dict] = []
    held_spans: set[str] = set()
    held: list[dict] = []
    trace = [{"step": "start", "risk": cur}]

    def _established(text: str) -> set[str]:
        fa, _, _ = check_faithful(faith_model, faith_check_prompt, text, core)
        return {t for t, kept in fa["points"] if kept}

    # Baseline: which core points the CURRENT message already establishes. The
    # backstop is DIFFERENTIAL — it holds a span only if generalizing it makes a
    # point go from established -> not established. Checking only the post-state
    # (as a first cut did) blames every generalization for any point the small
    # model can't verify at all (e.g. an inferred "wants fair compensation"),
    # which holds everything. Comparing against the baseline isolates the points
    # this specific generalization actually drops.
    kept_points = _established(working)

    def _try(c: dict, require_reduction: bool, reason: str) -> None:
        nonlocal working, cur, kept_points
        if c["span"] not in working or c["span"] in held_spans:
            return
        candidate_text = working.replace(c["span"], c["generic"])
        new = assess_local(reid_model, reid_prompt, candidate_text, brief=True)[0]["risk"]
        if require_reduction and _RANK.get(new, 99) >= _RANK.get(cur, 99):
            trace.append({"step": f"skip {c['span']!r} (no reduction)", "risk": new})
            return
        # Faithfulness backstop (differential): hold only if generalizing this drops
        # a point that WAS established — never trade the user's point for privacy.
        cand_points = _established(candidate_text)
        lost = [p for p in kept_points if p not in cand_points]
        if lost:
            held_spans.add(c["span"])
            held.append({"span": c["span"], "category": c["category"], "lost": lost})
            trace.append({"step": f"hold {c['span']!r} (would drop core: {lost})", "risk": cur})
            return
        working, cur, kept_points = candidate_text, new, cand_points
        applied.append(c)
        trace.append({"step": f"generalize {c['span']!r} ({reason})", "risk": new})

    # Floor pass — detected specific employers/locations are ALWAYS generalized
    # (subject only to task-function SPECIFIC and the faithfulness backstop), never
    # gated on the re-id score. The re-id proxy is too noisy to be the gate for
    # basic entity scrubbing: it rated "i'm <name> ... a company called praxis" as
    # low, which would otherwise leave a real company verbatim. Names + hard ids are
    # already floored deterministically upstream (_base_scrub); this extends the
    # floor to the soft named entities.
    floor = [c for c in gen if c["category"] in _FLOOR_CATEGORIES]
    combo = [c for c in gen if c["category"] not in _FLOOR_CATEGORIES]
    for c in floor:
        _try(c, require_reduction=False, reason="floor")

    # Pass 1 — reduction-driven (minimum-necessary) on the remaining combination
    # candidates.
    for c in combo:
        _try(c, require_reduction=True, reason="reduces risk")

    # Pass 2 — close the gap if still above target. Safe-by-construction: only
    # task-fn GENERAL + faithfulness-safe candidates, never below target.
    applied_spans = {c["span"] for c in applied}
    if _RANK.get(cur, 99) > _RANK.get(target_risk, 0):
        for c in combo:
            if _RANK.get(cur, 99) <= _RANK.get(target_risk, 0):
                break
            if c["span"] in applied_spans or c["span"] in held_spans:
                continue
            _try(c, require_reduction=False, reason="coverage")

    return {
        "task_function": {c["span"]: tf[c["span"]] for c in candidates},
        "applied": applied,
        "held": held,
        "search_protected": working,
        "search_final_risk": cur,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Item 4 — grammar-correct rendering via the existing rewriter
# ---------------------------------------------------------------------------


# Collapse accidental double articles the rewriter leaves when its generalization
# carries its own article next to the original's ("a a startup-…", "the an honor
# roll"). Keep the second article — it's the one attached to the new noun phrase.
_DUP_ARTICLE_RE = re.compile(r"\b(a|an|the)\s+(a|an|the)\b", re.IGNORECASE)


def _fix_articles(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _DUP_ARTICLE_RE.sub(lambda m: m.group(2), text)
    return text


def render_protected(
    base_message: str,
    applied_spans: list[dict],
    rewriter_model: str,
    rewriter_prompt: str,
) -> tuple[str, list[Replacement]]:
    """Produce the final protected message by generalizing `applied_spans` through
    the same rewriter the base pipeline uses — grammar-correct, not naive substitution.

    `base_message` already has names/hard-ids handled, so no name list is passed.
    Returns (protected_message, soft_replacements).
    """
    from .rewriter import build_substitutions_user_message, call_rewriter  # eval-only dep (lazy)

    if not applied_spans:
        return base_message, []
    generalize_spans = [{"span": s["span"], "category": s["category"]} for s in applied_spans]
    generalize_spans.sort(key=lambda s: base_message.find(s["span"]))
    user_msg, generalize_numbers = build_substitutions_user_message(base_message, {}, generalize_spans)
    rewriter_output, _ = call_rewriter(rewriter_model, rewriter_prompt, user_msg)
    protected, replacements = parse_rewriter_output(
        rewriter_output, {}, generalize_spans, generalize_numbers
    )

    # Applied-but-not-rendered fallback: the breaker DECIDED to generalize these
    # spans, but the 7B rewriter sometimes leaves one verbatim (empty/missing tag).
    # Substitute the span's crude generic deterministically — but ONLY when that
    # generic is a clean noun phrase ("a company", "a city"). The vague person
    # generics ("someone", "a person") mangle role words after "the"/"my"
    # ("the staff engineer level" -> "the someone"); for those, leaving the
    # low-value word verbatim (and surfacing it) reads far better than breaking the
    # sentence. _fix_articles then cleans any double article the substitution leaves.
    _VAGUE = {"someone", "a person"}
    rep_by_orig = {r.original: r for r in replacements}
    for s in applied_spans:
        span, generic = s["span"], s.get("generic", "someone")
        if span in protected and generic not in _VAGUE:
            protected = protected.replace(span, generic)
            if span in rep_by_orig:
                rep_by_orig[span].replacement = generic
            else:
                replacements.append(Replacement(original=span, replacement=generic, category=s["category"]))

    protected = _fix_articles(protected)
    return protected, replacements


def _classify_residual(
    detail: str,
    protected: str,
    core: dict,
    faith_model: str,
    faith_check_prompt: str,
) -> str:
    """Why a residual identifying detail was kept — core vs merely identifying.

    Probe: would generalizing this detail away drop one of the user's core points?
    If yes, it is the core (keep it, surface it, let the user decide); if no, it is
    identifying-but-not-central (a candidate the user may want to soften). This is
    uniform whether the detail was kept by task-function (SPECIFIC) or by the
    faithfulness backstop. Differential (like the backstop): compare what the
    message establishes before vs. after generalizing this detail, so a point the
    model simply can't verify isn't mistaken for one this detail carries.
    """
    from .faithfulness import check_faithful  # eval-only dep (lazy)

    loc = _locate_pin(detail, protected)
    if loc is None:
        return "identifying — kept"
    probe = protected.replace(loc, "someone")
    before = {t for t, k in check_faithful(faith_model, faith_check_prompt, protected, core)[0]["points"] if k}
    after = {t for t, k in check_faithful(faith_model, faith_check_prompt, probe, core)[0]["points"] if k}
    if before - after:
        return "core point — generalizing it would change your question"
    return "identifying but not central — consider softening"


# Categories that get a realistic FAKE of the same type (not a placeholder). Names
# are handled separately (gender-matched pseudonyms). Load-bearing content
# (medical/legal specifics, amounts, dates) is NOT faked — a fake diagnosis would
# change the answer — so it is kept (or handled by the slow breaker).
_FAKE_NONNAME = {"employer", "location", "email", "phone", "account_id", "secret", "address"}


# --- should-scrub gate: keep entities that are load-bearing or not the user's PII ---
# Detection != scrub-decision. The span finder flags any company/place/name-shaped token;
# this gate decides which to actually fake. We KEEP (do not fake): locations (low-attribution
# alone and usually load-bearing — the slow combination breaker handles identifying-in-
# combination cases); the subject of an informational question; and public companies/products
# the user is asking about or transacting with (not affiliated with). Bias: when framed, keep —
# faking a load-bearing / non-PII entity silently corrupts the answer, the worst failure mode.
_KEEP_FRAME_COMPANY = (
    "what is", "what's", "whats", "what are", "what does", "tell me about",
    "looking at", "look at", "considering", "evaluating", "thinking about",
    "should i use", "should i get", "should i switch", "should i buy", "compare",
)  # NB: no bare "vs" — it fires on "200k vs 180k at amazon" (salary, not product compare)
_KEEP_FRAME_NAME = (
    "the name", "the word", "what does", "what's", "whats", "who is", "who's",
    "meaning of", "etymology", "origin of", "pronounce", "spell",
)
_AFFILIATION = (
    "work at", "work for", "working at", "my company", "my employer", "my job",
    "we use", "employed at", "job at", "i work",
)
_ACCOUNT_NOUNS = (
    "account", "checking", "savings", "subscription", "plan", "policy",
    "membership", "card", "loan", "mortgage",
)
_KEEP_CATS = {"name", "employer", "other_identifier"}
# Categories routed to the fine-tuned relevance judge (when CLOAK_RELEVANCE_MODEL is set)
# instead of the keyword keep-gate. Location is the validated category; employer ~= CAPID's
# organization. Names stay on the keyword gate (the model scrubs ~all names; our gate keeps
# subject-of-question name frames). See harness/relevance.py + the 2026-06-16 exploration.
_RELEVANCE_CATS = {"location", "employer"}

# A location is load-bearing — keep it — only when it sits in a geographically-grounded
# task frame: a place-lookup ("hotel/university/store ... in X"), a place question
# ("what/where ... in X"), media/transit from a place ("watch ... from X"), or a stated
# current location with a local task ("I'm in X ... store/gift/near"). A narrative-backdrop
# location ("a wedding in Tuscany", "moving to Denver" amid a conflict) matches none of
# these and is scrubbed — it can be identifying-in-combination (the slow breaker handles
# those). A loose "geo-word anywhere" test over-keeps narrative locations badly (measured).
# Local-search place nouns only. Deliberately excludes vacation/housing-narrative words
# (rental, resort, beach, condo) — "a vacation rental in Cabo" is backdrop, not a lookup.
_PLACE_NOUN = (
    r"(hotel|motel|restaurant|store|shop|mall|bar|cafe|coffee|university|college|campus|"
    r"airport|museum|gym|clinic|hospital|attraction|tourist|things to do|nightlife|school)"
)
_LOC_SHOP_TASK = (
    "store", "shop", "gift", "buy", "near", "nearby", "local", "close", "open",
    "food", "coffee", "restaurant", "deliver", "pick up", "errand",
)
_LOC_MEDIA = ("watch", "stream", "fly", "flying", "flight", "drive", "driving", "broadcast", "vpn", "geo")
# `before` = text immediately preceding the location, lowercased, anchored at end ($).
# `(\w+\s+){0,2}$` allows adjectives / a second location token (e.g. "soho nyc").
_LOC_PLACE_RE = re.compile(_PLACE_NOUN + r"\b[^.?!]{0,30}\b(in|at|near|around|to)\s+(\w+\s+){0,2}$")
_LOC_WH_RE = re.compile(r"\b(what|which|where)\b[^.?!]{0,30}\b(in|at)\s+(\w+\s+){0,2}$")
_LOC_IM_IN_RE = re.compile(r"\b(i'?m|i am|im)\s+(in|at|near)\s+(\w+\s+){0,2}$")
_LOC_FROM_RE = re.compile(r"\bfrom\s+(\w+\s+){0,2}$")


def _location_load_bearing(text: str, message: str) -> bool:
    low = message.lower()
    t = text.lower()
    start = 0
    while (i := low.find(t, start)) >= 0:
        before = low[max(0, i - 60):i]
        if _LOC_PLACE_RE.search(before) or _LOC_WH_RE.search(before):
            return True
        if _LOC_IM_IN_RE.search(before) and any(w in low for w in _LOC_SHOP_TASK):
            return True
        if _LOC_FROM_RE.search(before) and any(w in low for w in _LOC_MEDIA):
            return True
        start = i + 1
    return False


_ADDR_GAP = re.compile(r"^[\s,.]*$")  # only connectors (space/comma/period) between fragments


def _coalesce_address_spans(message: str, spans: list[dict]) -> list[dict]:
    """A typed street address is detected as FRAGMENTS — '412 Broadway' (address),
    'New York' (location), 'NY' (location), '10012' (address) — and each gets its own fake,
    with the street and the zip each expanding into a full fake address: incoherent gibberish
    ('338 Hill St, Dayton, TX 75201, Provo, Austin 5147 Willow St, Tampa, WA 98101'). Merge a
    contiguous run of address/location spans (separated only by connectors) into ONE address
    span when the run contains a real address fragment, so it substitutes once as a coherent
    fake address. A bare city list (all 'location', no 'address') is NOT merged — those stay
    independent city fakes."""
    addr_loc = {"address", "location"}
    located: list[tuple[int, int, dict]] = []
    loose: list[dict] = []
    for s in spans:
        i = message.find(s["span"])
        if i >= 0:
            located.append((i, i + len(s["span"]), s))
        else:
            loose.append(s)
    located.sort()
    out = list(loose)
    j, n = 0, len(located)
    while j < n:
        si, ei, s = located[j]
        if s["category"] in addr_loc:
            run = [located[j]]
            run_end = ei
            k = j + 1
            while k < n:
                ns, ne, ns_s = located[k]
                if ns_s["category"] in addr_loc and _ADDR_GAP.match(message[run_end:ns]):
                    run.append(located[k])
                    run_end = ne
                    k += 1
                else:
                    break
            if len(run) > 1 and any(r[2]["category"] == "address" for r in run):
                out.append({"span": message[run[0][0]:run[-1][1]], "category": "address"})
                j = k
                continue
        out.append(s)
        j += 1
    return out


def _should_keep_span(text: str, category: str, message: str) -> bool:
    """True if this detected entity should be KEPT verbatim (not faked). Hard identifiers
    (email/phone/account_id/secret/address) are never kept here."""
    if os.environ.get("CLOAK_NO_KEEP_GATE"):
        return False  # ablation: disable the keep-gate to measure its impact
    if category == "location":
        # keep only when in a geographically-grounded task frame (not narrative backdrop)
        return _location_load_bearing(text, message)
    if category not in _KEEP_CATS:
        return False
    low = message.lower()
    t = text.lower()
    start = 0
    while (i := low.find(t, start)) >= 0:
        before = low[max(0, i - 40):i]
        if category == "name":
            if any(k in before for k in _KEEP_FRAME_NAME):
                return True
        elif not any(a in before for a in _AFFILIATION):
            if any(k in before for k in _KEEP_FRAME_COMPANY):
                return True
            if ("i have a" in before or "i have an" in before) and any(n in low for n in _ACCOUNT_NOUNS):
                return True
        start = i + 1
    return False


def fast_scrub(
    message: str,
    span_finder_model: str,
    span_prompt: str,
    existing: dict | None = None,
    profile=None,
    relevance_model: str | None = None,
) -> tuple[str, list[Replacement], dict]:
    """Floor scrub with realistic FAKE substitution — fast, natural, reversible.

    Every identifying entity is replaced with a plausible fake of the same type
    (name -> pseudonym, city -> a different city, email -> a fake email, code ->
    a same-shape code), so the protected message reads like a normal message and the
    cloud model can't tell anything changed. Deterministic string substitution, no
    rewriter and no re-id/faithfulness search — a single span-finder call.

    `existing` carries the conversation's map across turns for stable substitution:
    `{real: (fake, category)}`. Returns (protected_message, replacements, updated_map).
    """
    existing = dict(existing or {})
    spans = _normalize_spans(message, span_finder_model, span_prompt)
    # Profile layer (ADR 0007): the user's known entities are matched deterministically and
    # unioned into detection — this catches what the span finder misses (unfamiliar
    # employers, out-of-distribution inputs) and applies the profile's stable fake. The
    # span finder remains the net for novel first-mentions. Profile spans go through the
    # same keep-gate below, so a known entity that IS the question's subject is still kept.
    if profile is not None:
        detected = {s["span"].lower() for s in spans}
        for m in profile.match(message):
            if m["span"].lower() not in detected:
                spans.append({"span": m["span"], "category": m["category"]})
            if m.get("fake"):
                existing.setdefault(m["span"], (m["fake"], m["category"]))
    # Coalesce a fragmented street address into one span so it fakes coherently.
    spans = _coalesce_address_spans(message, spans)
    # should-scrub gate: drop spans that are load-bearing or not the user's PII so they
    # are kept verbatim (the subject of a question, a public company/product, a location).
    # For locations/orgs, the fine-tuned relevance judge decides keep vs scrub when enabled —
    # one question-aware Ollama call per message; it replaces the keyword keep-gate for those
    # categories. Any failure -> empty map -> keyword-gate fallback. The model is chosen by the
    # explicit `relevance_model` arg (the app passes config.relevance_model); when None we fall
    # back to CLOAK_RELEVANCE_MODEL so CLI/harness callers keep working. Pass "" to disable.
    _rel_model = relevance_model if relevance_model is not None else os.environ.get("CLOAK_RELEVANCE_MODEL")
    _model_keep = {}
    if _rel_model and not os.environ.get("CLOAK_NO_KEEP_GATE"):
        _judge_spans = [s["span"] for s in spans if s["category"] in _RELEVANCE_CATS]
        _model_keep = relevance.judge(message, _judge_spans, _rel_model)

    def _keep(s: dict) -> bool:
        if s["category"] in _RELEVANCE_CATS and s["span"] in _model_keep:
            return _model_keep[s["span"]]
        return _should_keep_span(s["span"], s["category"], message)

    spans = [s for s in spans if not _keep(s)]
    used = {fake for fake, _ in existing.values()}

    # A span also flagged as a fakeable entity is not a person name
    # ("a company called praxis" -> a fake company, not a fake name).
    nonname = {s["span"] for s in spans if s["category"] in _FAKE_NONNAME}
    name_spans = [
        s for s in spans
        if s["category"] == "name" and _looks_like_name(s["span"]) and s["span"] not in nonname
    ]
    # Collapse overlapping name detections (drop a name that is a substring of a longer
    # one) so they share one pseudonym.
    name_spans.sort(key=lambda s: len(s["span"]), reverse=True)
    deduped: list[dict] = []
    for s in name_spans:
        if not any(s["span"].lower() in k["span"].lower() for k in deduped):
            deduped.append(s)
    name_spans = deduped

    # Corruption guard: trim verbs/emotions/function words the detectors swallowed into
    # a name span ("brother mike owes" -> "mike"), and drop pure false positives
    # ("i'm frustrated" -> "" -> not a name at all). One fix for every detector.
    trimmed: list[dict] = []
    for s in name_spans:
        core = _name_core(s["span"])
        if core:
            trimmed.append({**s, "span": core})
    name_spans = trimmed

    # Names -> stable gender-matched pseudonyms (seeded from prior name assignments).
    name_existing = {real: fake for real, (fake, cat) in existing.items() if cat == "name"}
    name_subs = assign_pseudonyms([s["span"] for s in name_spans], message, existing=name_existing)
    # Component mapping: a multi-word name's tokens share its pseudonym, guarded against
    # common-word homonyms (will/may/mark).
    for full, pseudo in list(name_subs.items()):
        for tok in full.split():
            t = tok.strip(".,'’").lower()
            if len(t) >= 4 and t not in _NOT_A_NAME and t not in _COMMON_WORDS and t not in name_subs:
                name_subs[t] = pseudo
    used |= set(name_subs.values())

    mapping = dict(existing)
    for real, fake in name_subs.items():
        mapping[real] = (fake, "name")

    # Other identity categories -> realistic fakes (stable across turns).
    other_spans: list[dict] = []
    seen: set[str] = set()
    _CODE_CATS = {"account_id", "secret", "other_identifier"}
    for s in spans:
        if s["category"] in _FAKE_NONNAME and s["span"] not in seen:
            # A code/secret span the LLM hallucinated from plain words ("drowning")
            # has no digit/cap/separator — don't fake it into gibberish.
            if s["category"] in _CODE_CATS and not _is_codey(s["span"]):
                continue
            seen.add(s["span"])
            other_spans.append(s)
    for s in other_spans:
        real = s["span"]
        if real not in mapping:
            fake = synthetics.fake_for(s["category"], real, used)
            used.add(fake)
            mapping[real] = (fake, s["category"])

    base = message
    reps: list[Replacement] = []
    # Non-name entities FIRST (plain replace, longest first) so an email/address is
    # scrubbed before a name-component replacement could rewrite inside it.
    nonname_pairs = [(s["span"], mapping[s["span"]][0], s["category"]) for s in other_spans if s["span"] in mapping]
    for real, fake, cat in sorted(nonname_pairs, key=lambda x: len(x[0]), reverse=True):
        if real in base:
            base = base.replace(real, fake)
            reps.append(Replacement(original=real, replacement=fake, category=cat))
    # Names (word-boundary), longest first.
    for orig in sorted(name_subs, key=len, reverse=True):
        if re.search(rf"\b{re.escape(orig)}\b", base, flags=re.IGNORECASE):
            base = re.sub(rf"\b{re.escape(orig)}\b", name_subs[orig], base, flags=re.IGNORECASE)
            reps.append(Replacement(original=orig, replacement=name_subs[orig], category="name"))

    # Fail-closed backstop: scrub any hard identifier (email/phone/card/ssn/key/ip) that
    # survived detection, so the privacy floor never depends on the LLM span finder.
    base, sweep_reps = _sweep_hard_ids(base, used, mapping)
    reps.extend(sweep_reps)
    # Ages-in-context: deterministic catch for bare numbers the model misses.
    base, age_reps = _sweep_ages(base, used, mapping)
    reps.extend(age_reps)

    return base, reps, mapping


# Descriptor words the re-id proxy attaches to a pin ("Jacob Smith (name and
# specific role)") — not the distinctive content, so they don't count as "surviving".
_PIN_DESCRIPTORS = {
    "the", "a", "an", "of", "in", "at", "my", "on", "to", "and", "for", "with",
    "is", "their", "there", "has", "who", "that", "also", "it", "from",
    "name", "specific", "role", "company", "companies", "person", "individual",
}


def _pin_survives(pin: str, protected: str, threshold: float = 0.5) -> bool:
    """Does a flagged pin's distinctive content still appear in the protected text?

    Used to filter the "couldn't auto-handle" list: a pin whose distinctive words are
    gone (e.g. "Jacob Smith" -> "Liam", "Praxis" -> "a tech firm") WAS handled via
    another path and must not be surfaced as unhandled.
    """
    words = {w for w in re.findall(r"[a-z0-9]+", pin.lower()) if len(w) >= 2 and w not in _PIN_DESCRIPTORS}
    if not words:
        return False
    hay = protected.lower()
    return sum(1 for w in words if w in hay) / len(words) >= threshold


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def run_breaker(
    message: str,
    span_finder_model: str,
    reid_model: str,
    task_model: str,
    faith_model: str,
    rewriter_model: str,
    span_prompt: str,
    reid_prompt: str,
    task_prompt: str,
    faith_extract_prompt: str,
    faith_check_prompt: str,
    rewriter_prompt: str,
) -> dict:
    """Detect -> base-scrub -> break (with faithfulness backstop) -> render.

    Returns a fully-resolved result: the protected message, the replacement map,
    which spans were generalized vs held, raw/final re-id, the task-function
    verdicts, the user's core, and the step-by-step trace.
    """
    from .faithfulness import extract_core  # eval-only dep (lazy)
    from .reid_local import assess_local  # eval-only dep (lazy)

    # Detection and core-extraction are independent (both read only the raw message)
    # — run them concurrently.
    with ThreadPoolExecutor(max_workers=2) as ex:
        det_fut = ex.submit(
            detect_candidates, message, span_finder_model, span_prompt, reid_model, reid_prompt
        )
        core_fut = ex.submit(extract_core, faith_model, faith_extract_prompt, message)
        det = det_fut.result()
        core = core_fut.result()[0]
    base, name_subs, base_reps = _base_scrub(message, det["name_spans"], det["hard_spans"])

    decision = break_combination(
        base,
        det["candidates"],
        core=core,
        task_model=task_model,
        reid_model=reid_model,
        faith_model=faith_model,
        task_prompt=task_prompt,
        reid_prompt=reid_prompt,
        faith_check_prompt=faith_check_prompt,
    )

    protected, soft_reps = render_protected(
        base, decision["applied"], rewriter_model, rewriter_prompt
    )
    final_reid = assess_local(reid_model, reid_prompt, protected)[0]

    # Rendering is always grammar-correct (the rewriter, in context). We do NOT
    # fall back to crude template substitution: a category like other_identifier
    # spans roles/projects/awards/verbs, and a one-size generic ("someone")
    # mangles them ("we just raised" -> "we just someone"). The cost is the
    # rewriter occasionally under-generalizing (slightly less private, honestly
    # surfaced in the residual) — the safe direction for a tool whose whole point
    # is that the answer stays useful.

    # Floor/zone preview: the residual identifying details still present in the
    # protected message — what the user is asked to review. A detail remains
    # because generalizing it would drop the core (the un-breakable case), because
    # the answer needs it (task-function SPECIFIC), or because we couldn't anchor
    # it. We classify each so the user sees why, rather than silently keeping or
    # silently gutting.
    located = [pin for pin in final_reid.get("pins", []) if _locate_pin(pin, protected) is not None]
    if located:
        with ThreadPoolExecutor(max_workers=min(len(located), 4)) as ex:
            reasons = list(ex.map(
                lambda pin: _classify_residual(pin, protected, core, faith_model, faith_check_prompt),
                located,
            ))
        residual = [{"detail": p, "reason": r} for p, r in zip(located, reasons)]
    else:
        residual = []

    # Trust filter: only surface a pin as "couldn't auto-handle" if its distinctive
    # content actually survived in the protected message. Pins that were handled via
    # another path (name -> pseudonym, employer -> generic) must not look unhandled.
    unlocated = [p for p in det["unlocated_pins"] if _pin_survives(p, protected)]

    return {
        "message": message,
        "base_message": base,
        "protected": protected,
        "replacements": [r.to_dict() for r in (base_reps + soft_reps)],
        "raw_risk": det["reid"].get("risk"),
        "final_risk": final_reid.get("risk"),
        "final_pins": final_reid.get("pins"),
        "candidates": det["candidates"],
        "name_subs": name_subs,
        "applied": [c["span"] for c in decision["applied"]],
        "held": decision["held"],
        "residual_identifying": residual,
        "unlocated_pins": unlocated,
        "task_function": decision["task_function"],
        "core": core,
        "trace": decision["trace"],
    }
