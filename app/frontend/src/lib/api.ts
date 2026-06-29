// Typed client for the cloak backend (localhost FastAPI, session-scoped).
// - Browser demo: the backend serves this bundle, so call it same-origin ("").
// - Tauri shell: the page loads over a non-http(s) protocol, so target the local backend port.
// - Override either with VITE_API_BASE if the backend runs elsewhere.

import { getByokKeys } from "@/lib/byok"

const API_BASE =
  import.meta.env.VITE_API_BASE ??
  (typeof window !== "undefined" && window.location.protocol.startsWith("http")
    ? "" // served same-origin by the FastAPI backend
    : "http://127.0.0.1:8765") // native (Tauri) shell talks to the local backend

export interface Change {
  from: string
  to: string
  category: string
  from_profile?: boolean // this swap matched the user's saved profile (vs. model detection)
}

export type SendMode = "protected" | "edited" | "original"

export interface ScrubResult {
  original: string
  protected: string
  changes: Change[]
}

export interface HealthInfo {
  ok: boolean
  local_model: string
  relevance_model: string // keep-vs-scrub judge for places/orgs ("" = keyword-gate fallback)
  cloud_model: string
  span_prompt: string
  has_key: boolean
}

export interface SessionMeta {
  id: string
  title: string
  updated_at: number
  save_enabled: boolean
}

export interface TranscriptMessage {
  role: "user" | "assistant"
  text: string
  modelSaw?: string
  changes?: Change[]
  rehydrations?: Change[]
  mode?: SendMode
}

export interface SessionDetail {
  id: string
  title: string | null
  save_enabled: boolean
  model: string
  effort: string
  messages: TranscriptMessage[]
}

export type StreamEvent =
  | { type: "sent"; mode: string; model_saw: string }
  | { type: "delta"; answer: string }
  | { type: "done"; answer: string; answer_model_saw: string; rehydrations: Change[] }
  | { type: "error"; error: string }

export interface ProfileEntity {
  id: number
  surface: string
  variants: string[]
  category: string
  status: "candidate" | "confirmed" | "rejected"
  source: "detected" | "user_added"
  seen_count: number // messages it has appeared in
  seen_chats: number // distinct chats
}

export interface ProfileOverview {
  active: ProfileEntity[]
  pending: ProfileEntity[] // candidates that recurred enough to be "up for promotion"
  watching: ProfileEntity[] // candidates below the threshold — observed but not yet suggested
  pending_count: number
}

export interface EffortOption {
  value: string
  label: string
}

export interface ReasoningInfo {
  label: string // "Effort" | "Intelligence" | "Thinking"
  default: string // default option value
  options: EffortOption[]
}

export interface ModelInfo {
  id: string
  label: string
  provider: string
  available: boolean // its provider key is loaded
  reasoning: ReasoningInfo | null // per-model effort/intelligence/thinking control
}

export interface ProviderInfo {
  id: string
  label: string
  available: boolean
  key_env: string
}

export interface ModelCatalog {
  models: ModelInfo[]
  providers: ProviderInfo[]
  default: string
}

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error ?? `request failed (${res.status})`)
  }
  return res.json()
}

export async function health(): Promise<HealthInfo> {
  return jsonOrThrow(await fetch(`${API_BASE}/health`))
}

export async function getModels(): Promise<ModelCatalog> {
  return jsonOrThrow(await fetch(`${API_BASE}/models`))
}

export interface Settings {
  default_save_history: boolean
  learn_from_chats: boolean
  promote_threshold: number
  default_model: string
  default_effort: string
}

export async function getSettings(): Promise<Settings> {
  return jsonOrThrow(await fetch(`${API_BASE}/settings`))
}

export async function putSettings(patch: Partial<Settings>): Promise<Settings> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  )
}

export async function listSessions(): Promise<SessionMeta[]> {
  return jsonOrThrow(await fetch(`${API_BASE}/sessions`))
}

export async function createSession(): Promise<{
  id: string
  save_enabled: boolean
  model: string
  effort: string
}> {
  return jsonOrThrow(await fetch(`${API_BASE}/sessions`, { method: "POST" }))
}

export async function getSession(id: string): Promise<SessionDetail> {
  return jsonOrThrow(await fetch(`${API_BASE}/sessions/${id}`))
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" })
}

/** Erase all saved conversations + learned profile from disk (server-side). */
export async function wipeLocalData(): Promise<void> {
  await fetch(`${API_BASE}/local-data`, { method: "DELETE" })
}

export async function patchSession(
  id: string,
  patch: { title?: string; save_enabled?: boolean; model?: string; effort?: string },
): Promise<void> {
  await fetch(`${API_BASE}/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  })
}

// -- profile (the per-user entity set; ADR 0007/0008) -----------------------
export async function getProfile(): Promise<ProfileOverview> {
  return jsonOrThrow(await fetch(`${API_BASE}/profile`))
}

export async function addProfileEntity(
  surface: string,
  category: string,
  variants: string[] = [],
): Promise<ProfileEntity> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ surface, category, variants }),
    }),
  )
}

export async function promoteProfileEntity(id: number): Promise<void> {
  await fetch(`${API_BASE}/profile/${id}/promote`, { method: "POST" })
}

export async function updateProfileEntity(
  id: number,
  patch: { surface?: string; category?: string; variants?: string[] },
): Promise<void> {
  await fetch(`${API_BASE}/profile/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  })
}

export async function rejectProfileEntity(id: number): Promise<void> {
  await fetch(`${API_BASE}/profile/${id}`, { method: "DELETE" })
}

/** Step 1: preview. Returns the protected text + what changed; sends nothing. The user's
 *  BYOK provider keys ride along (local backend only) so the same key set is used end-to-end. */
export async function scrub(sessionId: string, message: string): Promise<ScrubResult> {
  const keys = getByokKeys()
  return jsonOrThrow(
    await fetch(`${API_BASE}/sessions/${sessionId}/scrub`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        anthropic_key: keys.anthropic ?? null,
        openai_key: keys.openai ?? null,
      }),
    }),
  )
}

/** Step 2: approved send. Streams SSE events; calls onEvent for each. */
export async function sendStream(
  sessionId: string,
  body: { mode: SendMode; text?: string; model?: string; effort?: string },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const keys = getByokKeys()
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...body,
      anthropic_key: keys.anthropic ?? null,
      openai_key: keys.openai ?? null,
    }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`send failed (${res.status})`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const line = frame.split("\n").find((l) => l.startsWith("data:"))
      if (!line) continue
      const json = line.slice(5).trim()
      if (json) onEvent(JSON.parse(json) as StreamEvent)
    }
  }
}
