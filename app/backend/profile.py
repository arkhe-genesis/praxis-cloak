"""Per-user PII profile store (app M3) — two buckets with promotion-on-recurrence.

ADR 0008 §3: the profile is the per-user ENTITY SET (detection), per-user; the fake is
per-session (sessions.map_blob), not stored here. This module adds the product model the
owner specified:

  - CANDIDATE bucket (status='candidate'): every soft entity the user approved-to-scrub is
    auto-observed here, counting how many MESSAGES it has appeared in (and which chats).
    Observation does NOT drive matching — it's just watching.
  - ACTIVE profile (status='confirmed'): entities PROMOTED from candidates once they've been
    mentioned enough (>= PROMOTE_THRESHOLD messages) and the user clicks Remember. ONLY these
    drive deterministic matching (build_profile). Manual adds go straight here.
  - status='rejected': the user dismissed/removed it; never observed or matched again.

Surfaces/variants AND the seen counts are PII, so they live in the encrypted content_blob
(crypto.py); category/status/source are structural metadata (plaintext). The profile is
small, so queries decrypt all rows — simple and fine at this scale.
"""

from __future__ import annotations

import time

from . import crypto, store
from . import harness_bridge as hb

_TABLE = """
CREATE TABLE IF NOT EXISTS profile_entities (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  category     TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'candidate',   -- candidate | confirmed | rejected
  source       TEXT NOT NULL DEFAULT 'detected',    -- detected | user_added
  created_at   REAL NOT NULL,
  updated_at   REAL NOT NULL,
  content_blob BLOB NOT NULL                          -- encrypted {surface, variants, count, seen_sessions}
);
"""

SOFT_CATEGORIES = {"name", "employer", "location", "other_identifier"}
PROMOTE_THRESHOLD = 2  # messages a candidate must appear in before it's "up for promotion"


def _name_components(surface: str) -> list[str]:
    """Component forms of a multi-word name ('Anna Smith' -> ['anna','smith']) so a later
    bare 'Anna' matches a profile entity first seen in full. Mirrors the >=4-char component
    logic in combine.fast_scrub / Profile.learn. Homonym components ('drew') are still gated
    at match time (corroboration required), so deriving them is safe."""
    toks = [t.strip(".,'’\"").lower() for t in surface.split()]
    return [t for t in toks if len(t) >= 4]


def _merge_forms(a: list[str], b: list[str]) -> list[str]:
    have = {x.lower() for x in a}
    return a + [x for x in b if x.lower() not in have]


class ProfileStore:
    def __init__(self) -> None:
        with store._conn() as c:
            c.executescript(_TABLE)

    # -- internal -----------------------------------------------------------
    def _all(self) -> list[dict]:
        with store._conn() as c:
            rows = c.execute(
                "SELECT id,category,status,source,content_blob FROM profile_entities ORDER BY id"
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            content = crypto.decrypt_json(r["content_blob"])
            seen_sessions = content.get("seen_sessions", [])
            out.append({
                "id": r["id"], "category": r["category"], "status": r["status"], "source": r["source"],
                "surface": content["surface"], "variants": content.get("variants", []),
                "seen_sessions": seen_sessions,
                "count": content.get("count", len(seen_sessions)),  # fallback for older rows
            })
        return out

    @staticmethod
    def _find(rows: list[dict], surface: str) -> dict | None:
        sl = surface.lower()
        return next((e for e in rows if e["surface"].lower() == sl), None)

    @staticmethod
    def _public(e: dict) -> dict:
        return {"id": e["id"], "surface": e["surface"], "variants": e["variants"],
                "category": e["category"], "status": e["status"], "source": e["source"],
                "seen_count": e["count"], "seen_chats": len(e["seen_sessions"])}

    def _write(self, c, entity_id, surface, variants, count, seen_sessions, status=None) -> None:
        blob = crypto.encrypt_json(
            {"surface": surface, "variants": variants, "count": count, "seen_sessions": seen_sessions})
        if status is None:
            c.execute("UPDATE profile_entities SET content_blob=?, updated_at=? WHERE id=?",
                      (blob, time.time(), entity_id))
        else:
            c.execute("UPDATE profile_entities SET content_blob=?, status=?, updated_at=? WHERE id=?",
                      (blob, status, time.time(), entity_id))

    # -- observation (candidate bucket) -------------------------------------
    def observe(self, session_id: str, entities: list[dict]) -> None:
        """Record soft entities sent scrubbed in `session_id` into the candidate bucket. Each
        call is one message: bump the mention count by 1 and note the chat. Names are
        consolidated so a person mentioned as 'Anna Smith' one message and 'Anna' the next
        counts as ONE candidate (see _observe_name); other categories match exact surface.
        Skips entities already confirmed (active) or rejected."""
        if not entities:
            return
        rows = self._all()
        now = time.time()
        with store._conn() as c:
            for e in entities:
                surface = (e.get("surface") or e.get("text") or "").strip()
                if not surface:
                    continue
                if e.get("category") == "name":
                    self._observe_name(c, rows, session_id, surface, now)
                else:
                    self._observe_simple(c, rows, session_id, surface, e["category"], now)

    def _insert_candidate(self, c, rows, surface, category, variants, count, seen, now) -> None:
        blob = crypto.encrypt_json(
            {"surface": surface, "variants": variants, "count": count, "seen_sessions": seen})
        cur = c.execute(
            "INSERT INTO profile_entities(category,status,source,created_at,updated_at,content_blob) "
            "VALUES(?,?,?,?,?,?)",
            (category, "candidate", "detected", now, now, blob))
        rows.append({"id": cur.lastrowid, "category": category, "status": "candidate",
                     "source": "detected", "surface": surface, "variants": variants,
                     "count": count, "seen_sessions": seen})

    def _bump(self, c, target, session_id, extra_count=0, extra_seen=()) -> None:
        count = target["count"] + 1 + extra_count
        seen = list(target["seen_sessions"])
        for s in [session_id, *extra_seen]:
            if s not in seen:
                seen.append(s)
        self._write(c, target["id"], target["surface"], target["variants"], count, seen)
        target["count"] = count
        target["seen_sessions"] = seen

    def _observe_simple(self, c, rows, session_id, surface, category, now) -> None:
        existing = self._find(rows, surface)
        if existing:
            if existing["status"] != "candidate":
                return  # confirmed or rejected -> leave alone
            self._bump(c, existing, session_id)
        else:
            self._insert_candidate(c, rows, surface, category, [], 1, [session_id], now)

    def _observe_name(self, c, rows, session_id, surface, now) -> None:
        """Consolidate fragmented name detections onto one canonical candidate. A bare
        first/last name folds into a multi-word candidate that contains it (only if exactly
        one does — otherwise it's ambiguous and stays separate); a multi-word name absorbs
        existing bare-component candidates. Conservative: only candidates are touched, and an
        ambiguous bare token is never attributed to a guessed person."""
        sl = surface.lower()
        exact = self._find(rows, surface)
        if exact and exact["status"] != "candidate":
            return  # a confirmed/rejected decision on this exact surface wins
        cands = [e for e in rows if e["status"] == "candidate" and e["category"] == "name"]
        comps = set(_name_components(surface))
        is_multi = len(surface.split()) > 1

        target = exact if (exact and exact["status"] == "candidate") else None
        absorb: list[dict] = []
        if is_multi:
            absorb = [e for e in cands if e is not target
                      and " " not in e["surface"].strip() and e["surface"].lower() in comps]
        elif target is None:
            containers = [e for e in cands if " " in e["surface"].strip()
                          and sl in set(_name_components(e["surface"]))]
            if len(containers) == 1:
                target = containers[0]

        extra_count = sum(f["count"] for f in absorb)
        extra_seen = [s for f in absorb for s in f["seen_sessions"]]
        if target:
            self._bump(c, target, session_id, extra_count, extra_seen)
        else:
            seen = list(dict.fromkeys([*extra_seen, session_id]))
            self._insert_candidate(c, rows, surface, "name", [], 1 + extra_count, seen, now)
        for f in absorb:
            c.execute("DELETE FROM profile_entities WHERE id=?", (f["id"],))
            rows.remove(f)

    # -- queries ------------------------------------------------------------
    def list_active(self) -> list[dict]:
        return [self._public(e) for e in self._all() if e["status"] == "confirmed"]

    def list_pending(self, threshold: int = PROMOTE_THRESHOLD) -> list[dict]:
        return [self._public(e) for e in self._all()
                if e["status"] == "candidate" and e["count"] >= threshold]

    def pending_count(self, threshold: int = PROMOTE_THRESHOLD) -> int:
        return len(self.list_pending(threshold))

    def list_watching(self, threshold: int = PROMOTE_THRESHOLD) -> list[dict]:
        """Candidates below the promotion threshold — observed but not yet 'up for
        promotion'. Surfaced so nothing stored is invisible (the user can dismiss any)."""
        return [self._public(e) for e in self._all()
                if e["status"] == "candidate" and e["count"] < threshold]

    def build_profile(self) -> "hb.Profile":
        ents = []
        for e in self._all():
            if e["status"] != "confirmed":
                continue
            variants = list(e["variants"])
            if e["category"] == "name":  # derive component forms so bare parts match
                variants = _merge_forms(variants, _name_components(e["surface"]))
            ents.append({"surface": e["surface"], "category": e["category"], "variants": variants})
        return hb.Profile.seeded(ents)

    # -- mutations ----------------------------------------------------------
    def add(self, surface: str, category: str, variants=()) -> dict:
        """Manual add — straight into the active profile (no recurrence needed)."""
        surface = surface.strip()
        variants = [v.strip() for v in variants if v.strip()]
        rows = self._all()
        existing = self._find(rows, surface)
        with store._conn() as c:
            if existing:
                have = {v.lower() for v in existing["variants"]}
                merged = existing["variants"] + [v for v in variants if v.lower() not in have]
                self._write(c, existing["id"], existing["surface"], merged, existing["count"],
                            existing["seen_sessions"], status="confirmed")
                return {**self._public(existing), "status": "confirmed", "variants": merged}
            now = time.time()
            blob = crypto.encrypt_json(
                {"surface": surface, "variants": variants, "count": 0, "seen_sessions": []})
            cur = c.execute(
                "INSERT INTO profile_entities(category,status,source,created_at,updated_at,content_blob) "
                "VALUES(?,?,?,?,?,?)",
                (category, "confirmed", "user_added", now, now, blob))
            return {"id": cur.lastrowid, "surface": surface, "variants": variants, "category": category,
                    "status": "confirmed", "source": "user_added", "seen_count": 0, "seen_chats": 0}

    def _set_status(self, entity_id: int, status: str) -> None:
        with store._conn() as c:
            c.execute("UPDATE profile_entities SET status=?, updated_at=? WHERE id=?",
                      (status, time.time(), entity_id))

    def promote(self, entity_id: int) -> None:
        self._set_status(entity_id, "confirmed")

    def reject(self, entity_id: int) -> None:
        """Dismiss a pending candidate OR remove an active entity — both -> rejected, so it
        is never re-observed or re-suggested (a hard delete would let it come back)."""
        self._set_status(entity_id, "rejected")

    def update(self, entity_id: int, *, surface=None, category=None, variants=None) -> dict | None:
        """Edit an entity in place: rename, recategorize, or change its variant/nickname list.
        Surface/variants are re-encrypted; category is structural."""
        e = next((x for x in self._all() if x["id"] == entity_id), None)
        if e is None:
            return None
        new_surface = surface.strip() if surface and surface.strip() else e["surface"]
        new_variants = (
            [v.strip() for v in variants if v.strip()] if variants is not None else e["variants"]
        )
        with store._conn() as c:
            self._write(c, entity_id, new_surface, new_variants, e["count"], e["seen_sessions"])
            if category:
                c.execute("UPDATE profile_entities SET category=?, updated_at=? WHERE id=?",
                          (category, time.time(), entity_id))
        return {**self._public(e), "surface": new_surface, "variants": new_variants,
                "category": category or e["category"]}
