"""Realistic synthetic substitutes — fakes, not placeholders.

Replace real PII with plausible fake values of the SAME TYPE/FORMAT, so the protected
message reads like a perfectly normal message and the 3rd-party model cannot tell
anything was changed (and so does not comment on "placeholders"). Reversible: the
caller keeps a real->fake map and rehydrates the model's reply back to the real values.

This extends ADR 0002's pseudonym-not-placeholder choice from names to every category.

Deterministic: the same real value seeds the same fake, so substitution is stable; the
caller passes a `used` set so distinct real values get distinct fakes (clean rehydration).
We do NOT fake load-bearing content (medical/legal specifics, amounts, dates) — a fake
diagnosis would change the answer. Those are kept or handled elsewhere.
"""

import hashlib
import re
import string

# Real, generic place names (a city is a city — a fake one reads naturally).
CITIES = [
    "Denver", "Austin", "Portland", "Charlotte", "Columbus", "Raleigh", "Tampa",
    "Sacramento", "Nashville", "Madison", "Boise", "Reno", "Tucson", "Albany",
    "Fresno", "Omaha", "Tulsa", "Mesa", "Aurora", "Spokane", "Richmond", "Akron",
    "Salem", "Provo", "Dayton", "Lincoln", "Tacoma", "Durham", "Modesto", "Erie",
]

# Invented-but-plausible company names (avoid real companies).
COMPANIES = [
    "Northwind", "Brightpath", "Meridian Labs", "Cobalt Systems", "Vanta Group",
    "Stillwater Co", "Larkspur", "Ironwood", "Greyline", "Halcyon", "Verana",
    "Kestrel", "Outpost", "Tidewater", "Brightwell", "Cardinal Works", "Lumen Co",
    "Atlas Forge", "Pinebrook", "Solstice Labs", "Harborview", "Nimbus", "Drift",
    "Quill", "Beacon Row", "Foundry 9", "Slate & Co", "Wayfare", "Orchard Park",
]

STREETS = [
    "Maple", "Oak", "Cedar", "Birch", "Elm", "Pine", "Walnut", "Chestnut",
    "Lincoln", "Lake", "Hill", "River", "Sunset", "Willow", "Aspen", "Linden",
    "Bridge", "Garden", "Spring", "Forest", "Meadow", "Highland", "Park", "Vine",
]

STATES = [
    ("CA", "94103"), ("NY", "10011"), ("TX", "75201"), ("WA", "98101"),
    ("CO", "80203"), ("IL", "60607"), ("OH", "43215"), ("GA", "30303"),
    ("OR", "97201"), ("AZ", "85003"), ("NC", "27601"), ("MA", "02118"),
]

FIRST_NAMES = [
    "alex", "sam", "jordan", "taylor", "casey", "riley", "morgan", "jamie", "drew",
    "avery", "quinn", "reese", "harper", "rowan", "emerson", "blake", "kai", "noa",
]
LAST_NAMES = [
    "morgan", "bennett", "carter", "hayes", "foster", "reed", "brooks", "porter",
    "sutton", "vance", "ellis", "shaw", "monroe", "doyle", "frost", "lane", "park",
]
EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com", "proton.me", "hotmail.com"]


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def _pick(pool: list[str], seed_str: str, used: set[str], avoid_real: str | None = None) -> str:
    """Deterministic pick from `pool`, skipping anything in `used`. `avoid_real` (the real
    value being faked) excludes any pool item that overlaps it, so a fake never echoes the
    real value — e.g. faking the city 'spokane' must not return 'Spokane' just because it's
    in the pool (the collision that leaked a real city inside a fake address)."""
    h = _seed(seed_str)
    n = len(pool)
    avoid = set(used)
    if avoid_real:
        rl = avoid_real.lower()
        avoid |= {c for c in pool if c.lower() in rl or rl in c.lower()}
    for i in range(n):
        cand = pool[(h + i) % n]
        if cand not in avoid:
            return cand
    return pool[h % n]


def _fake_digits(real: str, seed: str) -> str:
    """Replace each digit, preserve everything else (format-preserving)."""
    h = _seed(seed)
    out, i = [], 0
    for ch in real:
        if ch.isdigit():
            d = (h >> (3 * i)) % 10
            if i == 0 and d < 2:  # avoid an area code starting 0/1
                d += 2
            out.append(str(d))
            i += 1
        else:
            out.append(ch)
    return "".join(out)


def _fake_code(real: str) -> str:
    """Per-character class-preserving fake (Q7R9XZ -> K3M8PT, AA 4821 -> QR 7193)."""
    h = _seed("code:" + real)
    out, i = [], 0
    for ch in real:
        if ch.isdigit():
            out.append(str((h >> (4 * i)) % 10)); i += 1
        elif ch.isupper():
            out.append(string.ascii_uppercase[(h >> (4 * i)) % 26]); i += 1
        elif ch.islower():
            out.append(string.ascii_lowercase[(h >> (4 * i)) % 26]); i += 1
        else:
            out.append(ch)
    return "".join(out)


def _fake_email(real: str, used: set[str]) -> str:
    first = _pick(FIRST_NAMES, "e1:" + real, set())
    last = _pick(LAST_NAMES, "e2:" + real, set())
    dom = _pick(EMAIL_DOMAINS, "ed:" + real, set())
    local = real.split("@")[0]
    m = re.search(r"\d{1,4}$", local)  # keep a trailing digit cluster if present
    suffix = m.group(0) if m else ""
    cand = f"{first}.{last}{suffix}@{dom}"
    n = 0
    while cand in used:
        n += 1
        cand = f"{first}.{last}{suffix}{n}@{dom}"
    return cand


def _fake_address(real: str, used: set[str]) -> str:
    h = _seed("addr:" + real)
    num = 100 + (h % 8900)
    street = _pick(STREETS, "st:" + real, used, avoid_real=real)
    city = _pick(CITIES, "ac:" + real, used, avoid_real=real)
    st, zip_ = STATES[h % len(STATES)]
    return f"{num} {street} St, {city}, {st} {zip_}"


_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_HOLIDAYS = ["thanksgiving", "christmas", "easter", "halloween", "hanukkah", "diwali"]


def _fake_date(real: str) -> str:
    """Shift month/weekday/holiday names and fake digit runs — a plausible *different* date."""
    def repl_word(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        for lst in (_MONTHS, _DAYS, _HOLIDAYS):
            if low in lst:
                r = lst[(lst.index(low) + 4) % len(lst)]
                return r.capitalize() if w[:1].isupper() else r
        return w
    s = re.sub(r"[A-Za-z]+", repl_word, real)
    return _fake_digits(s, "dt:" + real)


def _fake_handle(real: str, used: set[str]) -> str:
    """A plausible social/messaging handle (@name + digits), preserving a separator."""
    base = _pick(FIRST_NAMES, "hn:" + real, set())
    h = _seed("hd:" + real)
    sep = "_" if "_" in real else ("." if "." in real else "")
    tail = _pick(LAST_NAMES, "hl:" + real, set()) if sep == "." else str(h % 90)
    cand = f"@{base}{sep}{tail}"
    n = 0
    while cand in used:
        n += 1
        cand = f"@{base}{sep}{tail}{n}"
    return cand


def _fake_dob(real: str) -> str:
    """A plausible but different birth date in the SAME format (valid month/day)."""
    h = _seed("dob:" + real)
    mo, day, yr = h % 12 + 1, (h >> 4) % 28 + 1, 1960 + (h >> 8) % 45
    if len(real) >= 4 and real[:4].isdigit() and "-" in real:  # YYYY-MM-DD
        return f"{yr:04d}-{mo:02d}-{day:02d}"
    sep = "/" if "/" in real else "-"
    return f"{mo:02d}{sep}{day:02d}{sep}{yr:04d}"


def _fake_age(real: str) -> str:
    """A plausible but different age, same digit count (54 -> a different adult age)."""
    h = _seed("age:" + real)
    try:
        n = int(real)
    except ValueError:
        return _fake_digits(real, "age:" + real)
    if n <= 17:  # a child's age -> a different child age 1-17
        return str((h % 17) + 1)
    return str((h % 62) + 18)  # adult 18-79


def fake_for(category: str, real: str, used: set[str]) -> str:
    """Return a realistic fake of the same type as `real`, distinct from `used`."""
    if category == "handle":
        return _fake_handle(real, used)
    if category == "dob":
        return _fake_dob(real)
    if category == "age":
        return _fake_age(real)
    if category == "location":
        return _pick(CITIES, "loc:" + real, used, avoid_real=real)
    if category == "employer":
        return _pick(COMPANIES, "emp:" + real, used, avoid_real=real)
    if category == "email":
        return _fake_email(real, used)
    if category == "phone":
        return _fake_digits(real, "ph:" + real)
    if category == "address":
        return _fake_address(real, used)
    if category in ("account_id", "secret", "other_identifier"):
        return _fake_code(real)
    if category == "exact_amount":
        return _fake_digits(real, "amt:" + real)  # similar magnitude, fake precision
    if category == "exact_date":
        return _fake_date(real)
    # Fallback: a plausible code (never a placeholder).
    return _fake_code(real)
