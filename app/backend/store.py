"""Local SQLite persistence (ADR 0008).

Stores conversations at the OS app-data dir. What lands in the DB:
  - sessions: id, created/updated, save_enabled, encrypted title, encrypted per-session
              pseudonym map (real->[fake,category]).
  - messages: role + seq (plaintext structure) + ENCRYPTED content ({scrubbed, mode}).
  - settings: small key/value (e.g. default_save_history).

Only structural metadata (ids, timestamps, roles, ordering) is plaintext; all content
and the maps are encrypted via crypto.py. Raw user text is never stored — only the
scrubbed transcript plus the encrypted map needed to rehydrate it on reload.

Persistence is whole-session snapshot (rewrite the session's messages on each save).
Conversations are small, so this stays simple; incremental append is a later optimization.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from . import crypto

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id           TEXT PRIMARY KEY,
  title_blob   BLOB,
  created_at   REAL NOT NULL,
  updated_at   REAL NOT NULL,
  save_enabled INTEGER NOT NULL DEFAULT 1,
  map_blob     BLOB,
  model        TEXT,
  effort       TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  seq          INTEGER NOT NULL,
  role         TEXT NOT NULL,
  content_blob BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def data_dir() -> Path:
    override = os.environ.get("CLOAK_DATA_DIR")
    p = Path(override) if override else Path.home() / "Library" / "Application Support" / "cloak"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return data_dir() / "cloak.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(db_path())
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        # migrate older DBs: add the per-session model/effort columns if missing
        cols = {r["name"] for r in c.execute("PRAGMA table_info(sessions)")}
        if "model" not in cols:
            c.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
        if "effort" not in cols:
            c.execute("ALTER TABLE sessions ADD COLUMN effort TEXT")


# -- settings ---------------------------------------------------------------
def get_setting(key: str, default: str | None = None) -> str | None:
    with _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


# -- sessions ---------------------------------------------------------------
def _subs_to_json(subs: dict) -> dict:
    return {real: [fake, cat] for real, (fake, cat) in subs.items()}


def _subs_from_json(d: dict) -> dict:
    return {real: (pair[0], pair[1]) for real, pair in d.items()}


def save_session(
    session_id: str,
    *,
    title: str | None,
    save_enabled: bool,
    subs: dict,
    records: list[dict],
    model: str | None = None,
    effort: str | None = None,
) -> None:
    """Upsert a session and replace its messages (whole-session snapshot)."""
    now = time.time()
    title_blob = crypto.encrypt_str(title) if title else None
    map_blob = crypto.encrypt_json(_subs_to_json(subs)) if subs else None
    with _conn() as c:
        prior = c.execute("SELECT created_at FROM sessions WHERE id=?", (session_id,)).fetchone()
        created = prior["created_at"] if prior else now
        c.execute(
            "INSERT INTO sessions(id,title_blob,created_at,updated_at,save_enabled,map_blob,model,effort) "
            "VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title_blob=excluded.title_blob, "
            "updated_at=excluded.updated_at, save_enabled=excluded.save_enabled, map_blob=excluded.map_blob, "
            "model=excluded.model, effort=excluded.effort",
            (session_id, title_blob, created, now, 1 if save_enabled else 0, map_blob, model, effort),
        )
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        c.executemany(
            "INSERT INTO messages(session_id,seq,role,content_blob) VALUES(?,?,?,?)",
            [
                (session_id, i, r["role"], crypto.encrypt_json({"scrubbed": r["scrubbed"], "mode": r.get("mode")}))
                for i, r in enumerate(records)
            ],
        )


def list_sessions() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id,title_blob,updated_at,save_enabled FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": crypto.decrypt_str(r["title_blob"]) if r["title_blob"] else "New conversation",
            "updated_at": r["updated_at"],
            "save_enabled": bool(r["save_enabled"]),
        }
        for r in rows
    ]


def load_session(session_id: str) -> dict | None:
    with _conn() as c:
        s = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not s:
            return None
        msgs = c.execute(
            "SELECT role,content_blob FROM messages WHERE session_id=? ORDER BY seq", (session_id,)
        ).fetchall()
    records = []
    for m in msgs:
        payload = crypto.decrypt_json(m["content_blob"])
        records.append({"role": m["role"], "scrubbed": payload["scrubbed"], "mode": payload.get("mode")})
    return {
        "id": session_id,
        "title": crypto.decrypt_str(s["title_blob"]) if s["title_blob"] else None,
        "save_enabled": bool(s["save_enabled"]),
        "subs": _subs_from_json(crypto.decrypt_json(s["map_blob"])) if s["map_blob"] else {},
        "records": records,
        "model": s["model"],
        "effort": s["effort"],
    }


def session_exists(session_id: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is not None


def rename_session(session_id: str, title: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET title_blob=?, updated_at=? WHERE id=?",
            (crypto.encrypt_str(title), time.time(), session_id),
        )


def delete_session(session_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))


def wipe_all() -> None:
    """Delete all user data — saved conversations and the learned profile. App preferences
    (the settings table) are kept; they hold no PII."""
    with _conn() as c:
        c.execute("DELETE FROM messages")
        c.execute("DELETE FROM sessions")
        c.execute("DELETE FROM profile_entities")
