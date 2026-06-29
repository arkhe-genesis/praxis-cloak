"""Profile-matching layer: deterministic detection of a user's known entities.

The per-user profile (ADR 0007) turns detection of recurring PII from open-vocabulary
*recognition* (hard for a small model) into *string matching* (free, ~perfect recall).
This module is the runtime match layer: given a profile of known entities, find every
occurrence in a message and emit spans in the SAME shape `combine._normalize_spans`
produces — so it slots in ahead of / alongside the fine-tuned span finder, which stays
the net for genuinely novel first-mentions.

Pure and model-free by design (that is the whole point — no inference for known entities).
The fake/keep decision is NOT made here: this layer only DETECTS; substitution and the
per-message task-function gate are downstream (see entity-decomposition.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _is_wordy(form: str) -> bool:
    """Purely alphabetic (incl. spaces/dots, e.g. 'dr. okafor') -> match on word
    boundaries so 'sam' doesn't fire inside 'same'. Emails/phones/codes use raw
    substring (they have their own delimiters)."""
    return all(c.isalpha() or c in " ." for c in form) and any(c.isalpha() for c in form)


# Name-homonyms that are also common English words. A profile entry of this form must
# NOT be blind-matched — 'may'/'art'/'drew' fire on the modal, the museum, the verb and
# corrupt the message (over-scrub). These forms require per-occurrence corroboration
# (the model co-fired on that span) instead of string presence. Starter list; extend.
_COMMON_WORDS: set[str] = {
    "may", "art", "drew", "will", "mark", "grace", "june", "rose", "summer", "hope",
    "faith", "dawn", "april", "rich", "bill", "jack", "ray", "norm", "sandy", "joy",
    "hazel", "ivy", "olive", "sky", "crystal", "pearl", "frank", "earnest",
}


def _is_ambiguous(form: str) -> bool:
    """A single-token form that is also a common word (only the bare variant — the
    multi-word 'drew haddad' is unambiguous, only 'drew' alone is)."""
    return " " not in form and form in _COMMON_WORDS


@dataclass
class Entity:
    surface: str
    category: str
    variants: list[str] = field(default_factory=list)
    fake: str | None = None        # the stable pseudonym (the real<->fake map; ADR 0002/0006)
    status: str = "confirmed"      # candidate (detected, unconfirmed) | confirmed | rejected
    source: str = "seeded"         # seeded | detected | user_added

    def forms(self) -> list[str]:
        forms = {self.surface.lower()}
        forms.update(v.lower() for v in self.variants)
        return sorted(forms, key=len, reverse=True)


@dataclass
class Profile:
    entities: list[Entity] = field(default_factory=list)
    _by_surface: dict[str, Entity] = field(default_factory=dict)

    @classmethod
    def seeded(cls, entity_dicts: list[dict]) -> "Profile":
        p = cls()
        for e in entity_dicts:
            p.add(e["surface"], e["category"], e.get("variants", []),
                  fake=e.get("fake"), status=e.get("status", "confirmed"),
                  source=e.get("source", "seeded"))
        return p

    def add(self, surface: str, category: str, variants=(), fake=None,
            status="confirmed", source="seeded") -> Entity:
        key = surface.lower()
        if key in self._by_surface:
            ent = self._by_surface[key]
            have = {v.lower() for v in ent.variants}
            for v in variants:
                if v.lower() not in have:
                    ent.variants.append(v)
            if fake and not ent.fake:
                ent.fake = fake
            return ent
        ent = Entity(surface=surface, category=category, variants=list(variants),
                     fake=fake, status=status, source=source)
        self.entities.append(ent)
        self._by_surface[key] = ent
        return ent

    def ensure_fakes(self, used: set[str] | None = None) -> "Profile":
        """Assign a stable fake to every confirmed entity that lacks one (deterministic,
        so the same real value always yields the same fake). Names get a gender-matched
        pseudonym (the same pool fast_scrub uses); other categories get a same-type fake.
        The app's ProfileStore persists these; here it also lets the harness run standalone."""
        from . import synthetics
        from .pseudonyms import assign_pseudonyms
        used = set(used or [])
        used |= {e.fake for e in self.entities if e.fake}
        names = [e for e in self.entities
                 if e.category == "name" and not e.fake and e.status != "rejected"]
        if names:
            subs = assign_pseudonyms([e.surface for e in names],
                                     " ".join(e.surface for e in names))
            for e in names:
                e.fake = subs.get(e.surface)
                if e.fake:
                    used.add(e.fake)
        for e in self.entities:
            if e.fake or e.status == "rejected":
                continue
            e.fake = synthetics.fake_for(e.category, e.surface, used)
            used.add(e.fake)
        return self

    def sub_map(self) -> dict[str, tuple[str, str]]:
        """Every known form -> (fake, category), for the substitution map. Skips
        unconfirmed/rejected entities and those without a fake."""
        out: dict[str, tuple[str, str]] = {}
        for e in self.entities:
            if not e.fake or e.status != "confirmed":
                continue
            for f in [e.surface, *e.variants]:
                out[f] = (e.fake, e.category)
        return out

    def learn(self, surface: str, category: str) -> Entity:
        """Accumulation hook: record an entity seen in a turn so later turns match it
        for free. For multi-word names, derive component variants (so a later bare
        'priya' matches an entity first seen as 'priya raman') — mirrors the component
        logic in combine.fast_scrub."""
        variants: list[str] = []
        if category == "name":
            toks = [t.strip(".,'’\"").lower() for t in surface.split()]
            variants = [t for t in toks if len(t) >= 4]
        return self.add(surface, category, variants)

    def _all_forms(self) -> list[tuple[str, Entity]]:
        out = [(f, ent) for ent in self.entities if ent.status != "rejected"
               for f in ent.forms()]
        out.sort(key=lambda x: len(x[0]), reverse=True)  # longest first; claim greedily
        return out

    def match(
        self,
        message: str,
        corroborated_ranges: list[tuple[int, int]] | None = None,
        gate_homonyms: bool = True,
    ) -> list[dict]:
        """All non-overlapping occurrences of known entity forms, longest form first.

        Ambiguous (common-word) forms — 'may', 'art', 'drew' — would blind-match the
        modal/verb/noun and corrupt the message, so when `gate_homonyms` is on they fire
        ONLY at occurrences that overlap a `corroborated_ranges` span (e.g. where the
        model also flagged a name). With no corroboration they don't fire — the safe
        default: defer homonym names to the model rather than over-scrub. Non-ambiguous
        forms always blind-match (the profile's whole point)."""
        low = message.lower()
        corr = corroborated_ranges or []
        claimed = [False] * len(message)
        spans: list[dict] = []
        for form, ent in self._all_forms():
            ambiguous = gate_homonyms and _is_ambiguous(form)
            start = 0
            while True:
                i = low.find(form, start)
                if i < 0:
                    break
                j = i + len(form)
                if _is_wordy(form):
                    lb = i == 0 or not low[i - 1].isalnum()
                    rb = j == len(low) or not low[j].isalnum()
                    if not (lb and rb):
                        start = i + 1
                        continue
                if any(claimed[i:j]):
                    start = j
                    continue
                if ambiguous and not any(a < j and i < b for a, b in corr):
                    start = j
                    continue
                for k in range(i, j):
                    claimed[k] = True
                spans.append({
                    "span": message[i:j], "category": ent.category,
                    "surface": ent.surface, "fake": ent.fake, "start": i, "end": j,
                })
                start = j
        return spans
