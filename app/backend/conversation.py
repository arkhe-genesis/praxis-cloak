"""In-memory state for one conversation (the chat engine; persistence is layered on
top by SessionManager).

The two-step trust loop:
  1. scrub(message)    -> protected text + what changed, stashed as `pending`; sends nothing.
  2. send_stream(mode) -> commit the scrub map, append the turn, stream the cloud reply,
                          rehydrate it back to the user's real entities.

State is held as `records` (each {role, scrubbed, mode}) — the scrubbed transcript, which
is exactly what gets persisted — plus the per-session `subs` map (real -> (fake, category)).
The cloud message list is derived from `records`. Pseudonyms are stable within the session
(and a resumed session reuses its stored map); a different session gets fresh fakes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

from . import harness_bridge as hb
from .transport import ChatTransport


class Conversation:
    def __init__(
        self,
        sid: str,
        local_model: str,
        span_prompt: str,
        transport: ChatTransport,
        save_enabled: bool = True,
        profile=None,
        model: str = "",
        effort: str = "",
        relevance_model: str = "",
    ):
        self.id = sid
        self.local_model = local_model
        self.relevance_model = relevance_model  # keep-vs-scrub judge for locations/orgs ("" = keyword gate)
        self.span_prompt = span_prompt
        self.transport = transport
        self.save_enabled = save_enabled
        self.profile = profile  # per-user entity set (harness Profile); detection only
        self.model = model  # cloud model this chat is on (per-session memory)
        self.effort = effort  # reasoning value for that model
        self.title: Optional[str] = None
        self.records: list[dict] = []                  # {role, scrubbed, mode}
        self.subs: dict = {}                            # real -> (fake, category), per session
        self.sub_map: dict[str, hb.Replacement] = {}    # real -> Replacement, cumulative
        self.pending: Optional[dict] = None
        self.last_observed: list[dict] = []             # soft entities sent last turn (for the profile)

    def load_state(
        self,
        *,
        subs: dict,
        records: list[dict],
        title: Optional[str],
        model: str = "",
        effort: str = "",
    ) -> None:
        self.subs = dict(subs)
        self.sub_map = {real: hb.Replacement(real, fake, cat) for real, (fake, cat) in subs.items()}
        self.records = list(records)
        self.title = title
        if model:
            self.model = model
            self.effort = effort or ""

    def _cloud_messages(self) -> list[dict]:
        return [{"role": r["role"], "content": r["scrubbed"]} for r in self.records]

    # -- step 1: preview ----------------------------------------------------
    _SOFT_CATEGORIES = {"name", "employer", "location", "other_identifier"}

    def _known_forms(self) -> set:
        known: set[str] = set()
        if self.profile is not None:
            for e in self.profile.entities:
                known |= set(e.forms())
        return known

    def scrub(self, message: str) -> dict:
        protected, reps, subs_after = hb.fast_scrub(
            message, self.local_model, self.span_prompt, existing=self.subs, profile=self.profile,
            relevance_model=self.relevance_model or None,
        )
        known = self._known_forms()
        candidates = self._new_candidates(reps, known)  # soft entities not yet in the profile
        self.pending = {"raw": message, "protected": protected, "reps": reps,
                        "subs_after": subs_after, "candidates": candidates}
        return {
            "original": message,
            "protected": protected,
            "changes": [
                {"from": r.original, "to": r.replacement, "category": r.category,
                 "from_profile": r.original.lower() in known}
                for r in reps
            ],
        }

    def _new_candidates(self, reps: list, known: set) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for r in reps:
            key = r.original.lower()
            if r.category in self._SOFT_CATEGORIES and key not in known and key not in seen:
                seen.add(key)
                out.append({"text": r.original, "category": r.category, "fake": r.replacement})
        return out

    # -- step 2: approved send ---------------------------------------------
    def _commit(self, reps: list, subs_after: dict) -> None:
        self.subs = subs_after
        for r in reps:
            self.sub_map[r.original] = r

    def send_stream(self, mode: str, edited_text: Optional[str]) -> Iterator[dict]:
        pending = self.pending or {}
        if mode == "original":
            to_send = pending.get("raw") or (edited_text or "")
            reps, commit = [], False
        elif mode == "edited":
            to_send = (edited_text or "").strip() or pending.get("protected") or ""
            reps, commit = pending.get("reps", []), True
        else:  # "protected"
            to_send = pending.get("protected") or (edited_text or "")
            reps, commit = pending.get("reps", []), True

        if not to_send.strip():
            yield {"type": "error", "error": "nothing to send"}
            return

        if commit and pending:
            self._commit(reps, pending["subs_after"])
            # Observe only entities whose fake actually survives in what we sent — so an edit
            # that removed or changed a substitution isn't learned from (edited mode).
            self.last_observed = [
                c for c in pending.get("candidates", []) if c.get("fake") and c["fake"] in to_send
            ]
        else:
            self.last_observed = []

        self.records.append({"role": "user", "scrubbed": to_send, "mode": mode})
        yield {"type": "sent", "mode": mode, "model_saw": to_send}

        all_reps = list(self.sub_map.values())
        acc = ""
        try:
            for chunk in self.transport.stream(self._cloud_messages()):
                acc += chunk
                shown = hb.rehydrate(acc, all_reps).rehydrated_response if all_reps else acc
                yield {"type": "delta", "answer": shown}
        except Exception as e:
            self.records.pop()  # roll back the user turn we appended for the failed call
            yield {"type": "error", "error": str(e)}
            return

        self.records.append({"role": "assistant", "scrubbed": acc, "mode": None})
        if all_reps:
            result = hb.rehydrate(acc, all_reps)
            final = result.rehydrated_response
            rehydrations = [
                {"from": e.replacement, "to": e.original, "category": e.category}
                for e in result.events
                if e.outcome == "substituted"
            ]
        else:
            final, rehydrations = acc, []
        self.pending = None
        yield {"type": "done", "answer": final, "answer_model_saw": acc, "rehydrations": rehydrations}
