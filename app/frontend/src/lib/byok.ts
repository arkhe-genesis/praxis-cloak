// BYOK (Bring Your Own Key) manager for local provider key storage.
//
// Keys live in localStorage ONLY and are sent solely to the user's own local backend
// (alongside scrub/send requests) — never to any other host. This is the open-source,
// no-auth path: the user supplies their own Anthropic / OpenAI key.

const ANTHROPIC_KEY_STORE = "gk_anthropic_key"
const OPENAI_KEY_STORE = "gk_openai_key"

export interface ByokKeys {
  anthropic?: string
  openai?: string
}

/** Read the stored provider keys. Empty/blank values are treated as unset. */
export function getByokKeys(): ByokKeys {
  try {
    return {
      anthropic: localStorage.getItem(ANTHROPIC_KEY_STORE)?.trim() || undefined,
      openai: localStorage.getItem(OPENAI_KEY_STORE)?.trim() || undefined,
    }
  } catch {
    return {}
  }
}

export function setAnthropicKey(key: string | null): void {
  const v = key?.trim()
  if (v) localStorage.setItem(ANTHROPIC_KEY_STORE, v)
  else localStorage.removeItem(ANTHROPIC_KEY_STORE)
}

export function setOpenaiKey(key: string | null): void {
  const v = key?.trim()
  if (v) localStorage.setItem(OPENAI_KEY_STORE, v)
  else localStorage.removeItem(OPENAI_KEY_STORE)
}

export function clearAllKeys(): void {
  localStorage.removeItem(ANTHROPIC_KEY_STORE)
  localStorage.removeItem(OPENAI_KEY_STORE)
}
