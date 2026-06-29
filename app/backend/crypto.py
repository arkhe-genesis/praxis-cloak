"""Local at-rest encryption (ADR 0008).

A single random 256-bit data key encrypts all sensitive payloads at rest — message
content and per-session pseudonym maps. The key lives in the OS keychain (via `keyring`)
and is held in memory only; structural metadata (ids, timestamps, roles) is not
encrypted. AES-256-GCM (AEAD) with a fresh random 96-bit nonce per blob, so a tampered
blob fails to decrypt. This same encrypt/decrypt is what will later produce the E2EE
blobs for cloud sync (ADR 0008 §6).

Headless/testing override: set CLOAK_CONTENT_KEY to a base64 32-byte key to bypass
the keychain (e.g. CI, or non-interactive shells where the keychain is locked).
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_SERVICE = "org.praxisnation.cloak"
_KEY_NAME = "local-content-key"


@lru_cache(maxsize=1)
def _key() -> bytes:
    env = os.environ.get("CLOAK_CONTENT_KEY")
    if env:
        return base64.b64decode(env)
    import keyring

    stored = keyring.get_password(_SERVICE, _KEY_NAME)
    if stored:
        return base64.b64decode(stored)
    key = AESGCM.generate_key(bit_length=256)
    keyring.set_password(_SERVICE, _KEY_NAME, base64.b64encode(key).decode())
    return key


def encrypt(plaintext: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(_key()).encrypt(nonce, plaintext, None)


def decrypt(blob: bytes) -> bytes:
    return AESGCM(_key()).decrypt(blob[:12], blob[12:], None)


def encrypt_str(s: str) -> bytes:
    return encrypt(s.encode("utf-8"))


def decrypt_str(blob: bytes) -> str:
    return decrypt(blob).decode("utf-8")


def encrypt_json(obj) -> bytes:
    return encrypt(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def decrypt_json(blob: bytes):
    return json.loads(decrypt(blob).decode("utf-8"))
