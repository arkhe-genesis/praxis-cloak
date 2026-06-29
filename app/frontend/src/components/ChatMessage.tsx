import { useState } from "react"
import { Check, Copy } from "lucide-react"
import type { ChatMsg } from "@/lib/types"
import { categoryLabel } from "@/lib/categories"
import { HighlightedText } from "@/components/HighlightedText"
import { Markdown } from "@/components/Markdown"

function Dots() {
  return (
    <span className="inline-flex items-center gap-1 py-1">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted [animation-delay:150ms]" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted [animation-delay:300ms]" />
    </span>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }
  return (
    <button
      type="button"
      onClick={copy}
      aria-label={copied ? "Copied" : "Copy response"}
      title={copied ? "Copied" : "Copy response"}
      className="flex items-center gap-1 text-muted transition-colors hover:text-foreground"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-protected" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  )
}

export function ChatMessage({ msg }: { msg: ChatMsg }) {
  if (msg.role === "user") {
    const changes = msg.changes ?? []
    // Highlight the real values in the user's own message; hovering a highlight reveals
    // what the model actually saw in its place, plus the entity type. User text is shown
    // verbatim (not markdown) — it's exactly what they typed.
    const marks = changes.map((c) => ({
      needle: c.from,
      tip: `${c.from} → ${c.to} · ${categoryLabel(c.category)}${c.from_profile ? " · from your profile" : ""}`,
      profile: c.from_profile,
    }))
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%]">
          <div className="whitespace-pre-wrap rounded-2xl rounded-br-sm bg-surface-2 px-4 py-2.5 text-[15px] leading-relaxed">
            <HighlightedText
              text={msg.text}
              marks={marks}
              markClassName="cursor-help underline decoration-protected/60 decoration-dotted underline-offset-2"
            />
          </div>
          {msg.mode === "original" && (
            <div className="mt-1 text-right text-[11px] text-danger/80">sent unprotected</div>
          )}
        </div>
      </div>
    )
  }

  // Assistant: no bubble (mirrors major LLM apps). Rendered as markdown; entities restored
  // locally are highlighted inline — hover reveals what the model wrote ("Mira → Anna").
  const rehy = msg.rehydrations ?? []
  const marks = rehy.map((r) => ({
    needle: r.to,
    tip: `${r.from} → ${r.to} · ${categoryLabel(r.category)}`,
  }))
  return (
    <div className="group/msg w-full">
      {msg.error ? (
        <div className="rounded-xl border border-danger/40 bg-surface px-4 py-2.5 text-[15px]">
          <span className="text-danger">{msg.error}</span>
        </div>
      ) : msg.text ? (
        <>
          <Markdown text={msg.text} marks={marks} />
          {!msg.streaming && (
            <div className="mt-1.5 flex items-center gap-2 opacity-0 transition-opacity group-hover/msg:opacity-100">
              <CopyButton text={msg.text} />
            </div>
          )}
        </>
      ) : (
        <Dots />
      )}
    </div>
  )
}
