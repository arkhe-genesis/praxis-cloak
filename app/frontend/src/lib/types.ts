import type { Change, SendMode } from "@/lib/api"

/** A single rendered chat message (UI state, not an API shape). */
export interface ChatMsg {
  id: string
  role: "user" | "assistant"
  text: string // displayed: user's real text, or the rehydrated assistant reply
  modelSaw?: string // provenance: what the cloud actually saw / wrote (scrubbed)
  changes?: Change[] // user msg: the substitutions applied (from=real, to=fake)
  rehydrations?: Change[] // assistant msg: restored entities (from=fake model wrote, to=real shown)
  mode?: SendMode // how the user msg was sent
  streaming?: boolean // assistant reply in progress
  error?: string
}
