"""Session orchestration: live conversations + persistence (ADR 0008).

Holds in-memory Conversation objects keyed by session id, backed by the local store.
Persists a session (whole-session snapshot) after a successful turn when save_enabled.
On load, rebuilds the displayed transcript from the stored scrubbed records + the
per-session map:
  - user messages   -> reverse ALL substitutions (reconstruct what you typed)
  - assistant replies-> apply the rehydration policy (what you saw)
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from typing import Optional

from . import harness_bridge as hb
from . import providers
from . import store
from .config import Config
from .conversation import Conversation
from .profile import ProfileStore
from .transport import ChatTransport

_SAVE_KEY = "default_save_history"
_LEARN_KEY = "learn_from_chats"
_THRESHOLD_KEY = "promote_threshold"
_MODEL_KEY = "default_model"
_EFFORT_KEY = "default_effort"


def _reverse_all(text: str, subs: dict) -> str:
    """Undo every fake->real substitution (longest fake first) to reconstruct the original."""
    for real, (fake, _cat) in sorted(subs.items(), key=lambda kv: len(kv[1][0]), reverse=True):
        if fake and fake in text:
            text = text.replace(fake, real)
    return text


def render_transcript(records: list[dict], subs: dict) -> list[dict]:
    reps = [hb.Replacement(real, fake, cat) for real, (fake, cat) in subs.items()]
    out: list[dict] = []
    for r in records:
        if r["role"] == "user":
            changes = [
                {"from": real, "to": fake, "category": cat}
                for real, (fake, cat) in subs.items()
                if fake and fake in r["scrubbed"]
            ]
            out.append(
                {
                    "role": "user",
                    "text": _reverse_all(r["scrubbed"], subs),
                    "modelSaw": r["scrubbed"],
                    "changes": changes,
                    "mode": r.get("mode"),
                }
            )
        else:
            res = hb.rehydrate(r["scrubbed"], reps)
            rehydrations = [
                {"from": e.replacement, "to": e.original, "category": e.category}
                for e in res.events
                if e.outcome == "substituted"
            ]
            out.append(
                {
                    "role": "assistant",
                    "text": res.rehydrated_response,
                    "modelSaw": r["scrubbed"],
                    "rehydrations": rehydrations,
                }
            )
    return out


class SessionManager:
    def __init__(self, cfg: Config, span_prompt: str):
        self.cfg = cfg
        self.span_prompt = span_prompt
        self._live: dict[str, Conversation] = {}
        self._transports: dict[str, ChatTransport] = {}  # cached per model|effort|keys
        self._profile_lock = threading.Lock()
        store.init_db()
        # The per-user PII profile (ADR 0007/0008): one shared harness Profile, mutated in
        # place on confirm/reject so every live conversation uses the latest entity set.
        self.profile_store = ProfileStore()
        self.profile = self.profile_store.build_profile()

    # -- settings -----------------------------------------------------------
    def default_save(self) -> bool:
        return store.get_setting(_SAVE_KEY, "true") == "true"

    def set_default_save(self, enabled: bool) -> None:
        store.set_setting(_SAVE_KEY, "true" if enabled else "false")

    def learn_enabled(self) -> bool:
        return store.get_setting(_LEARN_KEY, "true") == "true"

    def set_learn_enabled(self, enabled: bool) -> None:
        store.set_setting(_LEARN_KEY, "true" if enabled else "false")

    def promote_threshold(self) -> int:
        try:
            return max(1, int(store.get_setting(_THRESHOLD_KEY, "2")))
        except (TypeError, ValueError):
            return 2

    def set_promote_threshold(self, n: int) -> None:
        store.set_setting(_THRESHOLD_KEY, str(max(1, int(n))))

    def default_model(self) -> str:
        """The model new chats start on. Falls back to the provider default if unset or if
        the saved id is no longer in the catalog (e.g. a model was retired)."""
        stored = store.get_setting(_MODEL_KEY)
        if stored and providers.is_known_model(stored):
            return stored
        return providers.default_model(self.cfg)

    def default_effort(self) -> str:
        """The reasoning value new chats start on, validated against the default model's
        options (so a stale value can't seed a broken request)."""
        model = self.default_model()
        valid = providers.effort_values(model)
        stored = store.get_setting(_EFFORT_KEY)
        if stored and (not valid or stored in valid):
            return stored
        return providers.default_effort(model) or ""

    def set_default_model(self, model: str, effort: str | None) -> None:
        if providers.is_known_model(model):
            store.set_setting(_MODEL_KEY, model)
        store.set_setting(_EFFORT_KEY, effort or "")

    def settings(self) -> dict:
        return {
            "default_save_history": self.default_save(),
            "learn_from_chats": self.learn_enabled(),
            "promote_threshold": self.promote_threshold(),
            "default_model": self.default_model(),
            "default_effort": self.default_effort(),
        }

    # -- models / transports ------------------------------------------------
    def models(self) -> dict:
        """Catalog for the composer's model picker (which providers have keys, etc.)."""
        return providers.catalog(self.cfg)

    def _transport_for(
        self,
        model_id: str | None,
        effort: str | None = None,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
    ) -> ChatTransport:
        """Resolve (and cache) the transport for a (model, effort) pair. BYOK: the per-request
        keys are part of the cache key, so a transport built with one user's key is never
        reused for another's (and a key change rebuilds the client)."""
        mid = model_id or providers.default_model(self.cfg)
        eff = effort if effort is not None else providers.default_effort(mid)
        key = f"{mid}|{eff}|{anthropic_key}|{openai_key}"
        t = self._transports.get(key)
        if t is None:
            t = providers.build_transport(
                mid, eff, self.cfg, anthropic_key=anthropic_key, openai_key=openai_key
            )
            self._transports[key] = t
        return t

    # -- lifecycle ----------------------------------------------------------
    def _new_conversation(self, sid: str, save_enabled: bool) -> Conversation:
        return Conversation(
            sid=sid,
            local_model=self.cfg.local_model,
            relevance_model=self.cfg.relevance_model,
            span_prompt=self.span_prompt,
            transport=self._transport_for(None),  # default; overridden per send by model+effort
            save_enabled=save_enabled,
            profile=self.profile,
            model=self.default_model(),  # new chats start on the configured default
            effort=self.default_effort(),
        )

    def get_or_load(self, sid: str) -> Conversation:
        if sid in self._live:
            return self._live[sid]
        data = store.load_session(sid)
        convo = self._new_conversation(sid, data["save_enabled"] if data else self.default_save())
        if data:
            convo.load_state(
                subs=data["subs"], records=data["records"], title=data["title"],
                model=data.get("model") or "", effort=data.get("effort") or "",
            )
        self._live[sid] = convo
        return convo

    def create(self) -> dict:
        sid = str(uuid.uuid4())
        convo = self._new_conversation(sid, self.default_save())
        self._live[sid] = convo
        return {"id": sid, "save_enabled": convo.save_enabled, "model": convo.model, "effort": convo.effort}

    def set_model(self, sid: str, model: Optional[str], effort: Optional[str]) -> dict:
        """Remember a chat's model/effort (picker change). Updates the live conversation; the
        new value rides along on the next save (we don't create a DB row just for this, so
        empty new chats don't litter the sidebar)."""
        convo = self.get_or_load(sid)
        if model is not None:
            convo.model = model
        if effort is not None:
            convo.effort = effort
        if convo.save_enabled and convo.records:
            self._persist(convo)
        return {"model": convo.model, "effort": convo.effort}

    def list(self) -> list[dict]:
        return store.list_sessions()

    def transcript(self, sid: str) -> dict:
        convo = self.get_or_load(sid)
        return {
            "id": sid,
            "title": convo.title,
            "save_enabled": convo.save_enabled,
            "model": convo.model,
            "effort": convo.effort,
            "messages": render_transcript(convo.records, self._clean_map(convo)),
        }

    def delete(self, sid: str) -> None:
        store.delete_session(sid)
        self._live.pop(sid, None)

    def rename(self, sid: str, title: str) -> None:
        convo = self._live.get(sid)
        if convo:
            convo.title = title
        if store.session_exists(sid):
            store.rename_session(sid, title)

    def set_save(self, sid: str, enabled: bool) -> None:
        convo = self.get_or_load(sid)
        convo.save_enabled = enabled
        if enabled:
            self._persist(convo)
        else:
            store.delete_session(sid)  # turning off deletes the stored copy (ADR 0008 §5)

    # -- profile (per-user entity set; ADR 0007/0008) -----------------------
    def profile_overview(self) -> dict:
        """Everything the Profile view needs in one call: the active entity set, the
        candidates up for promotion, and the alert-dot count."""
        t = self.promote_threshold()
        pending = self.profile_store.list_pending(t)
        return {
            "active": self.profile_store.list_active(),
            "pending": pending,
            "watching": self.profile_store.list_watching(t),
            "pending_count": len(pending),
        }

    def profile_add(self, surface: str, category: str, variants=()) -> dict:
        rec = self.profile_store.add(surface, category, variants)
        self._refresh_profile()
        return rec

    def profile_update(self, entity_id: int, *, surface=None, category=None, variants=None) -> dict | None:
        rec = self.profile_store.update(entity_id, surface=surface, category=category, variants=variants)
        self._refresh_profile()
        return rec

    def profile_promote(self, entity_id: int) -> None:
        self.profile_store.promote(entity_id)
        self._refresh_profile()

    def profile_reject(self, entity_id: int) -> None:
        self.profile_store.reject(entity_id)
        self._refresh_profile()

    def _refresh_profile(self) -> None:
        """Rebuild the profile from the store and hand the new snapshot to every live
        conversation. We swap the reference (atomic) instead of mutating internals, under a
        lock so concurrent profile edits serialize; in-flight scrubs keep the snapshot they
        already hold."""
        with self._profile_lock:
            self.profile = self.profile_store.build_profile()
            for convo in self._live.values():
                convo.profile = self.profile

    def wipe_local(self) -> None:
        """Erase all on-disk user data (conversations + learned profile) and reset in-memory
        state. App preferences are kept. Used by the "Clear all local data" control."""
        with self._profile_lock:
            store.wipe_all()
            self._live.clear()
            self._transports.clear()
            self.profile = self.profile_store.build_profile()  # rebuilt empty

    # -- chat ---------------------------------------------------------------
    def scrub(
        self,
        sid: str,
        message: str,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
    ) -> dict:
        # Keys are accepted for a consistent BYOK contract with send_stream(); scrub itself
        # never makes a cloud call, so they're not used until the approved send.
        return self.get_or_load(sid).scrub(message)

    def send_stream(
        self,
        sid: str,
        mode: str,
        text: Optional[str],
        model: Optional[str] = None,
        effort: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
    ) -> Iterator[dict]:
        convo = self.get_or_load(sid)
        convo.transport = self._transport_for(
            model, effort, anthropic_key=anthropic_key, openai_key=openai_key
        )  # honor the composer's model+effort pick and the user's BYOK keys
        if model is not None:  # remember what this chat is on (per-session memory)
            convo.model = model
            convo.effort = effort or ""
        ok = False
        for event in convo.send_stream(mode, text):
            if event.get("type") == "done":
                ok = True
            yield event
        if ok:
            if convo.save_enabled:
                self._persist(convo)
            # observe the soft entities we just sent scrubbed -> the candidate bucket, so
            # they can recur toward promotion. Independent of save-history (the profile is
            # its own cross-session asset, fully visible/controllable in the Profile view).
            if convo.last_observed and self.learn_enabled():
                self.profile_store.observe(sid, convo.last_observed)

    # -- persistence --------------------------------------------------------
    def _clean_map(self, convo: Conversation) -> dict:
        """The deduped real->(fake, category) map — one entry per real, no lowercase
        component duplicates (which would create ambiguous origins and break rehydration
        on reload). Derived from sub_map, the same source the live rehydration uses."""
        return {r.original: (r.replacement, r.category) for r in convo.sub_map.values()}

    def _persist(self, convo: Conversation) -> None:
        if convo.title is None:
            convo.title = self._derive_title(convo)
        store.save_session(
            convo.id, title=convo.title, save_enabled=True, subs=self._clean_map(convo),
            records=convo.records, model=convo.model, effort=convo.effort,
        )

    def _derive_title(self, convo: Conversation) -> str:
        clean = self._clean_map(convo)
        for r in convo.records:
            if r["role"] == "user":
                real = _reverse_all(r["scrubbed"], clean).strip()
                first = real.splitlines()[0] if real else ""
                return (first[:48] + "…") if len(first) > 48 else (first or "New conversation")
        return "New conversation"
