import { useEffect, useRef, useState, type KeyboardEvent } from "react"
import { ArrowUp, Pencil, X } from "lucide-react"
import type { ModelCatalog, ScrubResult, SendMode } from "@/lib/api"
import type { ByokKeys } from "@/lib/byok"
import { categoryLabel } from "@/lib/categories"
import { Button } from "@/components/ui/button"
import { HighlightedText } from "@/components/HighlightedText"
import { ModelPicker } from "@/components/ModelPicker"

/** The input area, including the inline review that grows out of the composer.
 *
 * Keyboard: Enter sends (or, when a review is showing, approves the scrubbed version —
 * i.e. a second Enter approves); Shift+Enter is a newline; Escape dismisses the review.
 */
export function ComposerArea({
  draft,
  onDraftChange,
  onSubmitDraft,
  preview,
  busy,
  onApprove,
  onSendOriginal,
  onCancelReview,
  catalog,
  model,
  effort,
  onSelectModel,
  byok,
  autoApprove,
  onClearAutoApprove,
}: {
  draft: string
  onDraftChange: (v: string) => void
  onSubmitDraft: () => void
  preview: ScrubResult | null
  busy: boolean
  onApprove: (mode: SendMode, text: string, always: boolean) => void
  onSendOriginal: () => void
  onCancelReview: () => void
  catalog: ModelCatalog | null
  model: string
  effort: string
  onSelectModel: (model: string, effort: string) => void
  byok: ByokKeys
  autoApprove: boolean
  onClearAutoApprove: () => void
}) {
  const taRef = useRef<HTMLTextAreaElement>(null)
  const editRef = useRef<HTMLTextAreaElement>(null)
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState("")

  // Reset edit state when a new preview arrives or clears.
  useEffect(() => {
    setEditing(false)
    setEditText(preview?.protected ?? "")
  }, [preview])

  useEffect(() => {
    if (editing) editRef.current?.focus()
  }, [editing])

  // Auto-grow the composer.
  useEffect(() => {
    const el = taRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [draft])

  function approve(always: boolean) {
    onApprove(editing ? "edited" : "protected", editing ? editText : (preview?.protected ?? ""), always)
  }

  function composerKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.nativeEvent.isComposing) return
    if (e.key === "Escape" && preview) {
      e.preventDefault()
      onCancelReview()
      return
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (busy) return
      if (preview) approve(false)
      else onSubmitDraft()
    }
  }

  function editKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.nativeEvent.isComposing) return
    if (e.key === "Escape") {
      e.preventDefault()
      setEditing(false)
      return
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onApprove("edited", editText, false)
    }
  }

  const fakeMarks = (preview?.changes ?? []).map((c) => ({
    needle: c.to,
    tip: `${c.from} → ${c.to} · ${categoryLabel(c.category)}${c.from_profile ? " · from your profile" : ""}`,
    profile: c.from_profile,
  }))

  const sendDisabled = busy || (!preview && !draft.trim())

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-4">
      {autoApprove && (
        <div className="mb-2 flex justify-center">
          <button
            type="button"
            onClick={onClearAutoApprove}
            className="rounded-full bg-surface-2 px-2.5 py-0.5 text-[11px] text-muted transition-colors hover:text-foreground"
            title="Auto-approving scrubbed messages this session — click to turn off"
          >
            auto-approve on · click to stop
          </button>
        </div>
      )}
      <div className="overflow-hidden rounded-2xl border border-border bg-surface-2 shadow-lg">
        {preview && (
          <div className="border-b border-border px-3 py-2.5">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[11px] font-medium uppercase tracking-wide text-muted">
                Scrubbed
              </span>
              <button
                type="button"
                onClick={onCancelReview}
                aria-label="Cancel review"
                className="text-muted transition-colors hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {editing ? (
              <textarea
                ref={editRef}
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                onKeyDown={editKeyDown}
                rows={2}
                className="w-full resize-y rounded-md border border-border bg-background/60 p-2 text-[14px] leading-relaxed focus:border-accent focus:outline-none"
              />
            ) : (
              <div className="flex items-start gap-2">
                <p className="flex-1 whitespace-pre-wrap text-[14px] leading-relaxed">
                  <HighlightedText
                    text={preview.protected}
                    marks={fakeMarks}
                    markClassName="cursor-help rounded bg-protected/15 px-0.5 text-protected"
                  />
                </p>
                <button
                  type="button"
                  onClick={() => setEditing(true)}
                  aria-label="Edit scrubbed message"
                  className="mt-0.5 shrink-0 text-muted transition-colors hover:text-foreground"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            <div className="mt-2.5 flex flex-wrap items-center gap-2">
              <Button variant="secondary" size="sm" onClick={onSendOriginal}>
                Send original
              </Button>
              <Button variant="primary" size="sm" onClick={() => approve(false)}>
                Approve
              </Button>
              <Button variant="ghost" size="sm" onClick={() => approve(true)}>
                Always approve
              </Button>
            </div>
          </div>
        )}

        <div className="px-3 pt-2.5">
          <textarea
            ref={taRef}
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            onKeyDown={composerKeyDown}
            rows={1}
            placeholder="How can I help?"
            className="max-h-52 min-h-8 w-full resize-none bg-transparent text-[15px] leading-relaxed placeholder:text-muted focus:outline-none"
          />
          <div className="flex items-center justify-between pb-2 pt-1">
            <ModelPicker
              catalog={catalog}
              model={model}
              effort={effort}
              onSelect={onSelectModel}
              byok={byok}
              disabled={busy}
            />
            <button
              type="button"
              onClick={() => (preview ? approve(false) : onSubmitDraft())}
              disabled={sendDisabled}
              aria-label={preview ? "Approve and send" : "Send message"}
              className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-accent text-accent-foreground transition hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:opacity-40 disabled:hover:brightness-100"
            >
              <ArrowUp className="h-[18px] w-[18px]" strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
