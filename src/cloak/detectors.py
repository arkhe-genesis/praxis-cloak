import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|pk_[a-z]{4,}_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|gh[pso]_[A-Za-z0-9]{30,})\b"
)
CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Hard structured identifiers. Used in two places: Stage-1 detection AND the
# leak-detector pass in validate.py (these must never appear in a protected
# message), so keep this set to unambiguous identifiers only.
DETECTORS: dict[str, re.Pattern] = {
    "email": EMAIL,
    "phone": PHONE,
    "api_key": API_KEY,
    "card_number": CARD,
    "ssn": SSN,
}

# Bare numeric forms the LLM span finder reliably misses (ages stated as
# "8-year-old", custody/asset splits like "50/50"). Used ONLY for Stage-1 span
# detection in the decomposed pipeline — deliberately NOT in the validate.py
# leak pass, since these are softer patterns and a stray match in a generalized
# protected message should not be treated as a hard leak.
AGE = re.compile(r"\b\d{1,3}-year-old\b")
RATIO = re.compile(r"\b\d{1,2}/\d{1,2}\b")  # 50/50, 70/30; note: would also match an m/d date — none occur in v0 data

STAGE1_DETECTORS: dict[str, re.Pattern] = {
    "age": AGE,
    "ratio": RATIO,
}


def run_detectors(text: str) -> dict[str, list[str]]:
    return {name: pat.findall(text) for name, pat in DETECTORS.items()}


def run_stage1_detectors(text: str) -> dict[str, list[str]]:
    """Stage-1-only detectors (ages, ratios) — not part of the validate leak pass."""
    return {name: pat.findall(text) for name, pat in STAGE1_DETECTORS.items()}


# Reference / case / account numbers like "D-1-GN-24-00891": 3+ dash/slash
# segments. Filtered in code to those that contain BOTH a letter and a real
# digit run, so natural hyphenated phrases ("8-year-old", "mother-in-law") and
# ISO dates ("2024-06-03", no letters) are NOT matched. These are unique direct
# identifiers and must be handled deterministically, never via the LLM rewriter.
_STRUCTURED_ID_CANDIDATE = re.compile(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+){2,}\b")


# Entities named via "a company called X" / "my startup named X". The 7B span
# finder keys on capitalization and misses these in lowercase/informal input
# ("i run a company called praxis"). High-precision: requires a business-entity
# noun + called/named. The captured entity is its first token plus any following
# Capitalized tokens (so "Acme Health Systems" is whole, but the trailing clause
# in "called praxis and i ..." is not swallowed).
# Follow-on tokens of a multi-word entity, case-insensitive (so "baton rouge" /
# "northbridge analytics" aren't truncated), bounded by stopwords instead of by
# capitalization — people don't capitalize.
# Follow-on tokens of a multi-word entity (case-insensitive, stopword-bounded). Token
# class excludes "." so it can't absorb a period and run across a sentence boundary
# ("audience. Can").
_ENT_CONT = (
    r"(?:\s+(?!(?i:and|or|but|the|a|an|in|on|at|for|with|to|is|was|are|i|we|my|me|so|now|just|then|"
    r"because|since|that|this|here|there)\b)[A-Za-z][\w&\-]*){0,2}"
)

_CALLED_ENTITY = re.compile(
    r"(?i:\b(?:compan(?:y|ies)|start-?up|firm|business|employer|org|organi[sz]ation|"
    r"nonprofit|agency|fund|brand|app|product|platform|school|university|college))\s+"
    r"(?i:called|named)\s+"
    r"([A-Za-z][\w&.\-]*" + _ENT_CONT + r")"
)


def find_called_entities(text: str) -> list[str]:
    """Entities introduced by 'company/startup/... called|named X' (employer-like)."""
    return [m.group(1).strip(" .,!?") for m in _CALLED_ENTITY.finditer(text) if m.group(1).strip(" .,!?")]


# Casual/lowercase personal names the 7B span finder misses. Detected by INTRODUCTION
# CONTEXT ("i'm X", "my cofounder X", "X here") rather than a name gazetteer, so common
# words ("i will go", "may works") are not mis-flagged. Captures 1-2 tokens; the second
# token is taken only if it's not a stopword (so "charlie callinan and i" -> "charlie
# callinan", but "charlie and i" -> "charlie").
_NAME_STOP = (
    "and|but|so|or|the|a|an|to|at|in|on|of|my|who|that|here|from|with|is|was|i|we|he|"
    "she|they|it|am|are|im|will|can|just|really|also|too|now|then|today|when|how|what|"
    # common verbs/nouns that follow a name but are NOT part of it ("annie lives",
    # "annie's birthday") — never swallow these into the name span
    "lives|live|lived|works|work|worked|said|says|say|told|tell|tells|went|goes|go|came|"
    "come|comes|has|had|have|wants|want|needs|need|likes|like|loves|love|thinks|think|"
    "called|calls|asked|asks|made|make|makes|got|get|gets|gave|give|gives|took|take|takes|"
    "keeps|kept|moved|moving|started|starting|turns|turned|birthday|wedding|party|funeral|"
    "graduation|because|since|always|never|recently|usually"
)
_NAME_TOKEN = r"[A-Za-z][A-Za-z'’\-]+"
_CONTEXT_NAME = re.compile(
    r"(?:\bi'?m\b|\bi am\b|\bmy name is\b|\bname'?s\b|\bthis is\b|\bnamed\b|\bcall me\b|"
    r"\bmy (?:co-?founder|friend|colleague|co-?worker|boss|manager|wife|husband|partner|"
    r"fianc[eé]+|girlfriend|boyfriend|brother|sister|mom|mum|dad|son|daughter|roommate|"
    r"mentor|teammate|assistant|client|lawyer|doctor|therapist|neighbou?r))\s+"
    rf"({_NAME_TOKEN}(?:\s+(?!(?:{_NAME_STOP})\b){_NAME_TOKEN})?)",
    re.IGNORECASE,
)


# Words that follow "i'm / this is" but are NOT names — so "i'm traveling next" or
# "i'm not sure" isn't read as a name. Gerunds (-ing) are rejected by rule; this covers
# the rest (articles, adjectives, adverbs, fillers).
_NOT_A_NAME_FIRST = {
    "a", "an", "the", "not", "just", "really", "so", "very", "sure", "sorry", "fine",
    "good", "great", "okay", "ok", "here", "there", "now", "still", "also", "about",
    "from", "at", "in", "on", "with", "currently", "pretty", "quite", "actually",
    "basically", "gonna", "kind", "sort", "happy", "sad", "glad", "excited", "nervous",
    "new", "trying", "able", "afraid", "aware", "done", "back", "out", "over", "into",
    "looking", "hoping", "wondering", "thinking", "feeling", "going", "doing",
}


# Employer named after "work at/for X" / "job at X" — catches lowercase company names
# the 7B span finder misses ("i work at praxis"). Generic org words are rejected so
# "work at a startup" / "work from home" don't match; a named employer ("Stanford")
# does.
# "work at X" / "job at X" only — NOT "work for/with X", which over-matches benign
# phrases ("work for a living", "work for a non-technical audience").
_WORK_AT = re.compile(
    r"(?i:\b(?:works?|worked|working|job|employed)\s+at)\s+"
    r"(?i:a\s+|an\s+|the\s+|my\s+)?"
    r"([A-Za-z][\w&.\-]*" + _ENT_CONT + r")"
)
_WORKPLACE_STOP = {
    "home", "night", "day", "work", "times", "place", "office", "school", "college",
    "university", "hospital", "company", "startup", "firm", "business", "agency",
    "gym", "bank", "store", "restaurant", "nonprofit", "org", "team", "house",
}


def find_workplaces(text: str) -> list[str]:
    """Employers named via 'work at/for X' (lowercase-safe; generic words rejected)."""
    out: list[str] = []
    for m in _WORK_AT.finditer(text):
        ent = m.group(1).strip(" .,!?")
        first = ent.split()[0].lower() if ent else ""
        if not first or first in _WORKPLACE_STOP or first.endswith("ing"):
            continue
        out.append(ent)
    return out


def find_contextual_names(text: str) -> list[str]:
    """Personal names introduced by context ('i'm X', 'my cofounder X'). Lowercase-safe.

    Rejects gerunds and common words after the trigger so "i'm traveling next" /
    "i'm not sure" are not mistaken for a name.
    """
    out: list[str] = []
    for m in _CONTEXT_NAME.finditer(text):
        name = m.group(1).strip(" .,!?")
        name = re.sub(r"['’]s?$", "", name).strip()  # "annie's" -> "annie" (keep the 's in the text)
        if not name:
            continue
        first = name.split()[0].lower()
        if first.endswith("ing") or first in _NOT_A_NAME_FIRST:
            continue
        out.append(name)
    return out


# Confirmation / booking / reference / account codes introduced by context, plus
# flight numbers. The structured-id detector only catches dashed forms; these catch
# the bare-token forms ("confirmation code Q7R9XZ", "flight AA 4821") that leaked.
_CODE_CONTEXT = re.compile(
    r"(?i:\b(?:confirmation|booking|reservation|reference|record\s+locator|locator|pnr|"
    r"order|ticket|case|policy|account|member(?:ship)?|tracking|invoice|claim)\b"
    r"[\s:#]*(?:number|code|no\.?|id|num|#)?[\s:#]*)"
    r"([A-Za-z0-9][A-Za-z0-9\-]{3,})"
)
_FLIGHT_RE = re.compile(r"\bflight\s+([A-Za-z]{2}\s?\d{2,4})\b", re.IGNORECASE)


def find_codes(text: str) -> list[str]:
    """Bare confirmation/booking/account codes and flight numbers."""
    out: list[str] = []
    for m in _CODE_CONTEXT.finditer(text):
        tok = m.group(1).strip()
        # A real code has a digit, or is an all-caps token of decent length.
        if any(c.isdigit() for c in tok) or (tok.isupper() and len(tok) >= 5):
            out.append(tok)
    for m in _FLIGHT_RE.finditer(text):
        out.append(m.group(1).strip())
    return out


def find_structured_ids(text: str) -> list[str]:
    out: list[str] = []
    for m in _STRUCTURED_ID_CANDIDATE.findall(text):
        if not (any(c.isalpha() for c in m) and any(c.isdigit() for c in m)):
            continue
        segs = re.split(r"[-/]", m)
        digit_run = any(sum(c.isdigit() for c in s) >= 3 for s in segs)
        numeric_segs = sum(1 for s in segs if s.isdigit())
        if digit_run or numeric_segs >= 2:
            out.append(m)
    return out
