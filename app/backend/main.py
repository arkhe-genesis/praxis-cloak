"""FastAPI app exposing the cloak chat loop over localhost (M2: session-scoped).

  GET    /health                 — config + whether a cloud key is present
  GET    /models                 — cloud model catalog + which providers have keys
  GET    /settings               — { default_save_history }
  PUT    /settings               — set default_save_history
  GET    /sessions               — list saved sessions (most-recent first)
  POST   /sessions               — create a new (live) session
  GET    /sessions/{id}          — load: { id, title, save_enabled, messages[] } (rehydrated)
  DELETE /sessions/{id}          — delete a session
  PATCH  /sessions/{id}          — rename and/or toggle save_enabled
  POST   /sessions/{id}/scrub    — step 1: preview (protected text + changes); sends nothing
  POST   /sessions/{id}/send     — step 2: approved send; SSE stream of the rehydrated reply
  GET    /profile                — { active[], pending[], pending_count } for the Profile view
  POST   /profile                — manually add an entity straight to the active profile
  POST   /profile/{id}/promote   — promote a pending candidate into the active profile
  DELETE /profile/{id}           — reject (dismiss a pending candidate / remove an active one)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import harness_bridge as hb
from .config import load_config
from .sessions import SessionManager

cfg = load_config()
_SPAN_PROMPT = hb.span_prompt(cfg.span_prompt_name)

app = FastAPI(title="cloak backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

manager = SessionManager(cfg, _SPAN_PROMPT)


class ScrubRequest(BaseModel):
    message: str
    anthropic_key: Optional[str] = None  # BYOK: user's Anthropic key (env fallback if absent)
    openai_key: Optional[str] = None  # BYOK: user's OpenAI key (env fallback if absent)


class SendRequest(BaseModel):
    mode: str = "protected"  # protected | edited | original
    text: Optional[str] = None
    model: Optional[str] = None  # cloud model id chosen in the composer (defaults server-side)
    effort: Optional[str] = None  # reasoning effort/intelligence for that model (defaults server-side)
    anthropic_key: Optional[str] = None  # BYOK: user's Anthropic key (env fallback if absent)
    openai_key: Optional[str] = None  # BYOK: user's OpenAI key (env fallback if absent)


class SettingsRequest(BaseModel):
    default_save_history: Optional[bool] = None
    learn_from_chats: Optional[bool] = None
    promote_threshold: Optional[int] = None
    default_model: Optional[str] = None
    default_effort: Optional[str] = None


class PatchSessionRequest(BaseModel):
    title: Optional[str] = None
    save_enabled: Optional[bool] = None
    model: Optional[str] = None
    effort: Optional[str] = None


class ProfileEntityRequest(BaseModel):
    surface: str
    category: str
    variants: list[str] = []


class ProfilePatchRequest(BaseModel):
    surface: Optional[str] = None
    category: Optional[str] = None
    variants: Optional[list[str]] = None


def _sse(events: Iterable[dict]) -> StreamingResponse:
    def gen() -> Iterator[bytes]:
        try:
            for event in events:
                yield f"data: {json.dumps(event)}\n\n".encode()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n".encode()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "local_model": cfg.local_model,
        "relevance_model": cfg.relevance_model,  # keep-vs-scrub judge ("" = keyword gate fallback)
        "cloud_model": cfg.cloud_model,
        "span_prompt": cfg.span_prompt_name,
        "has_key": bool(cfg.anthropic_api_key),
    }


@app.get("/models")
def get_models() -> dict:
    return manager.models()


@app.get("/settings")
def get_settings() -> dict:
    return manager.settings()


@app.put("/settings")
def put_settings(req: SettingsRequest) -> dict:
    if req.default_save_history is not None:
        manager.set_default_save(req.default_save_history)
    if req.learn_from_chats is not None:
        manager.set_learn_enabled(req.learn_from_chats)
    if req.promote_threshold is not None:
        manager.set_promote_threshold(req.promote_threshold)
    if req.default_model is not None:
        manager.set_default_model(req.default_model, req.default_effort)
    return manager.settings()


@app.get("/sessions")
def list_sessions() -> list[dict]:
    return manager.list()


@app.post("/sessions")
def create_session() -> dict:
    return manager.create()


@app.get("/sessions/{sid}")
def get_session(sid: str) -> dict:
    return manager.transcript(sid)


@app.delete("/sessions/{sid}")
def delete_session(sid: str) -> dict:
    manager.delete(sid)
    return {"ok": True}


@app.delete("/local-data")
def wipe_local_data() -> dict:
    """Erase all saved conversations and the learned profile from this device.
    (The browser separately clears its stored API keys.)"""
    manager.wipe_local()
    return {"ok": True}


@app.patch("/sessions/{sid}")
def patch_session(sid: str, req: PatchSessionRequest) -> dict:
    if req.title is not None:
        manager.rename(sid, req.title)
    if req.save_enabled is not None:
        manager.set_save(sid, req.save_enabled)
    if req.model is not None or req.effort is not None:
        manager.set_model(sid, req.model, req.effort)
    return {"ok": True}


@app.post("/sessions/{sid}/scrub")
def scrub(sid: str, req: ScrubRequest):
    message = (req.message or "").strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    return JSONResponse(
        manager.scrub(
            sid, message, anthropic_key=req.anthropic_key, openai_key=req.openai_key
        )
    )


@app.post("/sessions/{sid}/send")
def send(sid: str, req: SendRequest) -> StreamingResponse:
    return _sse(
        manager.send_stream(
            sid,
            req.mode,
            req.text,
            req.model,
            req.effort,
            anthropic_key=req.anthropic_key,
            openai_key=req.openai_key,
        )
    )


@app.get("/profile")
def get_profile() -> dict:
    return manager.profile_overview()


@app.post("/profile")
def add_profile_entity(req: ProfileEntityRequest):
    surface = (req.surface or "").strip()
    if not surface:
        return JSONResponse({"error": "empty surface"}, status_code=400)
    return JSONResponse(manager.profile_add(surface, req.category, req.variants))


@app.patch("/profile/{entity_id}")
def update_profile_entity(entity_id: int, req: ProfilePatchRequest):
    rec = manager.profile_update(
        entity_id, surface=req.surface, category=req.category, variants=req.variants
    )
    if rec is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(rec)


@app.post("/profile/{entity_id}/promote")
def promote_profile_entity(entity_id: int) -> dict:
    manager.profile_promote(entity_id)
    return {"ok": True}


@app.delete("/profile/{entity_id}")
def reject_profile_entity(entity_id: int) -> dict:
    manager.profile_reject(entity_id)
    return {"ok": True}


# Serve the built React frontend. Mounted LAST so it never shadows the API routes above:
# the SPA (html=True) catches everything else and falls back to index.html for client-side
# routing. Skipped gracefully when the frontend hasn't been built yet.
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
