import { useState } from "react"
import { Check, Pencil, Plus, X } from "lucide-react"
import type { ProfileEntity, ProfileOverview } from "@/lib/api"
import { categoryLabel } from "@/lib/categories"
import { Button } from "@/components/ui/button"

const ADD_CATEGORIES = ["name", "employer", "location", "other_identifier"] as const

/** The per-user PII profile: the things cloak always protects. "Up for promotion" is
 * the candidate bucket — entities you've mentioned across enough chats to be worth
 * remembering; promote them into the active set (which is what actually drives matching). */
export function ProfileView({
  overview,
  onPromote,
  onReject,
  onAdd,
  onEdit,
}: {
  overview: ProfileOverview | null
  onPromote: (id: number) => void
  onReject: (id: number) => void
  onAdd: (surface: string, category: string) => void
  onEdit: (id: number, patch: { surface?: string; category?: string; variants?: string[] }) => void
}) {
  const pending = overview?.pending ?? []
  const watching = overview?.watching ?? []
  const active = overview?.active ?? []

  // group active entities by category for readable display
  const groups = new Map<string, ProfileEntity[]>()
  for (const e of active) {
    const arr = groups.get(e.category) ?? []
    arr.push(e)
    groups.set(e.category, arr)
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-2xl px-6 py-8">
        <h1 className="text-lg font-semibold">Your profile</h1>
        <p className="mt-1 text-sm text-muted">
          The names, employers, and places cloak always protects for you. Stored only on
          this device, encrypted — you control it.
        </p>

        <section className="mt-6 rounded-lg border border-border bg-surface px-4 py-4">
          <h2 className="text-sm font-medium">About you</h2>
          <p className="mb-3 mt-1 text-xs text-muted">
            Add the basics so cloak protects them from your very first message.
          </p>
          <div className="flex flex-col gap-2">
            <SeedField label="Your name" category="name" placeholder="" onAdd={onAdd} />
            <SeedField label="Where you work" category="employer" placeholder="" onAdd={onAdd} />
            <SeedField label="Where you live" category="location" placeholder="" onAdd={onAdd} />
          </div>
        </section>

        {pending.length > 0 && (
          <section className="mt-7">
            <h2 className="mb-1 text-sm font-medium">Up for promotion</h2>
            <p className="mb-3 text-xs text-muted">
              You’ve mentioned these across a few chats. Remember them and they’ll be protected
              reliably — even when the on-device detector would miss them.
            </p>
            <div className="flex flex-col gap-1.5">
              {pending.map((e) => (
                <div
                  key={e.id}
                  className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2"
                >
                  <span className="rounded bg-protected/15 px-1.5 py-0.5 text-sm text-protected">
                    {e.surface}
                  </span>
                  <span className="flex-1 text-xs text-muted">
                    {categoryLabel(e.category)} · mentioned {e.seen_count} times
                  </span>
                  <Button variant="primary" size="sm" onClick={() => onPromote(e.id)}>
                    <Check className="mr-1 h-3.5 w-3.5" /> Remember
                  </Button>
                  <button
                    type="button"
                    onClick={() => onReject(e.id)}
                    aria-label="Dismiss"
                    className="text-muted transition-colors hover:text-foreground"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        {watching.length > 0 && (
          <section className="mt-7">
            <h2 className="mb-1 text-sm font-medium">Also noticed</h2>
            <p className="mb-3 text-xs text-muted">
              Mentioned once so far — not protected yet, just being watched. Remember to protect
              now, or dismiss to stop tracking.
            </p>
            <div className="flex flex-col gap-1">
              {watching.map((e) => (
                <div
                  key={e.id}
                  className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-surface"
                >
                  <span className="text-sm">{e.surface}</span>
                  <span className="text-xs text-muted">· {categoryLabel(e.category)}</span>
                  <span className="flex-1" />
                  <button
                    type="button"
                    onClick={() => onPromote(e.id)}
                    className="text-xs text-muted opacity-0 transition-opacity hover:text-protected group-hover:opacity-100"
                  >
                    Remember
                  </button>
                  <button
                    type="button"
                    onClick={() => onReject(e.id)}
                    aria-label="Dismiss"
                    className="text-muted opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="mt-7">
          <h2 className="mb-3 text-sm font-medium">Always protected</h2>
          {active.length === 0 ? (
            <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-sm text-muted">
              Nothing yet. As you chat, the things you mention often will show up here to
              remember. You can also add one below.
            </p>
          ) : (
            <div className="flex flex-col gap-4">
              {[...groups.entries()].map(([cat, items]) => (
                <div key={cat}>
                  <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted">
                    {categoryLabel(cat)}
                  </div>
                  <div className="flex flex-col gap-1">
                    {items.map((e) => (
                      <ActiveEntityRow key={e.id} entity={e} onEdit={onEdit} onReject={onReject} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          <AddEntity onAdd={onAdd} />
        </section>
      </div>
    </div>
  )
}

function AddEntity({ onAdd }: { onAdd: (surface: string, category: string) => void }) {
  const [open, setOpen] = useState(false)
  const [surface, setSurface] = useState("")
  const [category, setCategory] = useState<string>("name")

  function submit() {
    const s = surface.trim()
    if (!s) return
    onAdd(s, category)
    setSurface("")
    setOpen(false)
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mt-3 flex items-center gap-1 text-sm text-muted transition-colors hover:text-foreground"
      >
        <Plus className="h-3.5 w-3.5" /> Add something to always protect
      </button>
    )
  }
  return (
    <div className="mt-3 flex items-center gap-2">
      <input
        autoFocus
        value={surface}
        onChange={(e) => setSurface(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder=""
        className="flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
      />
      <select
        value={category}
        onChange={(e) => setCategory(e.target.value)}
        className="rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
      >
        {ADD_CATEGORIES.map((c) => (
          <option key={c} value={c}>
            {categoryLabel(c)}
          </option>
        ))}
      </select>
      <Button variant="primary" size="sm" onClick={submit}>
        Add
      </Button>
      <button
        type="button"
        onClick={() => setOpen(false)}
        aria-label="Cancel"
        className="text-muted transition-colors hover:text-foreground"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}

function SeedField({
  label,
  category,
  placeholder,
  onAdd,
}: {
  label: string
  category: string
  placeholder: string
  onAdd: (surface: string, category: string) => void
}) {
  const [v, setV] = useState("")
  function submit() {
    const s = v.trim()
    if (!s) return
    onAdd(s, category)
    setV("")
  }
  return (
    <div className="flex items-center gap-2">
      <label className="w-28 shrink-0 text-sm text-muted">{label}</label>
      <input
        value={v}
        onChange={(e) => setV(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder={placeholder}
        className="flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
      />
      <Button variant="secondary" size="sm" onClick={submit} disabled={!v.trim()}>
        Add
      </Button>
    </div>
  )
}

function ActiveEntityRow({
  entity,
  onEdit,
  onReject,
}: {
  entity: ProfileEntity
  onEdit: (id: number, patch: { surface?: string; category?: string; variants?: string[] }) => void
  onReject: (id: number) => void
}) {
  const [editing, setEditing] = useState(false)
  const [surface, setSurface] = useState(entity.surface)
  const [category, setCategory] = useState(entity.category)
  const [variants, setVariants] = useState(entity.variants.join(", "))

  function save() {
    const s = surface.trim()
    if (!s) return
    onEdit(entity.id, {
      surface: s,
      category,
      variants: variants.split(",").map((x) => x.trim()).filter(Boolean),
    })
    setEditing(false)
  }

  if (editing) {
    const field = "rounded-md border border-border bg-background/60 px-2 py-1 text-sm focus:border-accent focus:outline-none"
    return (
      <div className="flex flex-wrap items-center gap-2 rounded-md bg-surface px-2 py-2">
        <input value={surface} onChange={(e) => setSurface(e.target.value)} className={`min-w-32 flex-1 ${field}`} />
        <select value={category} onChange={(e) => setCategory(e.target.value)} className={field}>
          {ADD_CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {categoryLabel(c)}
            </option>
          ))}
        </select>
        <input
          value={variants}
          onChange={(e) => setVariants(e.target.value)}
          placeholder="nicknames, comma-separated"
          className={`min-w-32 flex-1 ${field}`}
        />
        <Button variant="primary" size="sm" onClick={save}>
          Save
        </Button>
        <button type="button" onClick={() => setEditing(false)} aria-label="Cancel" className="text-muted hover:text-foreground">
          <X className="h-4 w-4" />
        </button>
      </div>
    )
  }

  return (
    <div className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-surface">
      <span className="text-sm">{entity.surface}</span>
      {entity.variants.length > 0 && (
        <span className="text-xs text-muted">· {entity.variants.join(", ")}</span>
      )}
      <span className="flex-1" />
      <button
        type="button"
        onClick={() => setEditing(true)}
        aria-label="Edit"
        className="text-muted opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
      >
        <Pencil className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => onReject(entity.id)}
        aria-label="Remove"
        className="text-muted opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
