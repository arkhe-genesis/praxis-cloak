import { type ReactNode, useState } from "react"
import { Eye, EyeOff } from "lucide-react"
import type { ModelCatalog } from "@/lib/api"
import type { ByokKeys } from "@/lib/byok"
import { Switch } from "@/components/ui/switch"
import { ModelPicker } from "@/components/ModelPicker"

/** App settings — chat/learning controls; built to gather more over time. */
export function SettingsView({
  defaultSave,
  onToggleDefaultSave,
  learnFromChats,
  onToggleLearn,
  promoteThreshold,
  onChangeThreshold,
  catalog,
  defaultModel,
  defaultEffort,
  onSelectDefaultModel,
  byok,
  onChangeAnthropicKey,
  onChangeOpenaiKey,
  spanModel,
  relevanceModel,
  onWipe,
}: {
  defaultSave: boolean
  onToggleDefaultSave: (v: boolean) => void
  learnFromChats: boolean
  onToggleLearn: (v: boolean) => void
  promoteThreshold: number
  onChangeThreshold: (n: number) => void
  catalog: ModelCatalog | null
  defaultModel: string
  defaultEffort: string
  onSelectDefaultModel: (model: string, effort: string) => void
  byok: ByokKeys
  onChangeAnthropicKey: (key: string) => void
  onChangeOpenaiKey: (key: string) => void
  spanModel?: string // on-device detector (finds the PII)
  relevanceModel?: string // on-device keep-vs-scrub judge ("" = keyword-gate fallback)
  onWipe: () => void
}) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-2xl px-6 py-8">
        <h1 className="text-lg font-semibold">Settings</h1>

        <section className="mt-7">
          <h2 className="mb-1 text-sm font-medium">Provider keys</h2>
          <p className="mb-3 text-xs text-muted">
            Bring your own key. Keys are stored on this device only (browser localStorage) and are
            sent solely to your local engine — never anywhere else.
          </p>
          <div className="flex flex-col gap-2">
            <KeyRow
              title="Anthropic"
              placeholder="sk-ant-…"
              value={byok.anthropic ?? ""}
              onChange={onChangeAnthropicKey}
            />
            <KeyRow
              title="OpenAI"
              placeholder="sk-…"
              value={byok.openai ?? ""}
              onChange={onChangeOpenaiKey}
            />
          </div>
        </section>

        <section className="mt-7">
          <h2 className="mb-3 text-sm font-medium">Chats</h2>
          <div className="flex flex-col gap-2">
            <Row
              title="Save new chats"
              desc="New conversations are kept locally (scrubbed transcript only). You can override per chat."
            >
              <Switch checked={defaultSave} onChange={onToggleDefaultSave} />
            </Row>
            <Row
              title="Learn recurring entities from chats"
              desc="Watch for names, employers, and places you mention often and suggest protecting them. Runs entirely on-device; turn off to stop observing. Review and manage everything learned in Profile."
            >
              <Switch checked={learnFromChats} onChange={onToggleLearn} />
            </Row>
            <Row
              title="Suggest after"
              desc="How many times you mention something before cloak suggests remembering it."
            >
              <Stepper value={promoteThreshold} onChange={onChangeThreshold} />
            </Row>
          </div>
        </section>

        <section className="mt-7">
          <h2 className="mb-3 text-sm font-medium">Cloud model</h2>
          <div className="flex flex-col gap-2">
            <Row
              title="Default for new chats"
              desc="Which cloud model and reasoning level new conversations start on. You can still switch per chat in the composer."
            >
              <ModelPicker
                catalog={catalog}
                model={defaultModel}
                effort={defaultEffort}
                onSelect={onSelectDefaultModel}
                byok={byok}
                side="bottom"
              />
            </Row>
          </div>
        </section>

        {(spanModel || relevanceModel !== undefined) && (
          <section className="mt-7">
            <h2 className="mb-3 text-sm font-medium">On-device models</h2>
            <div className="flex flex-col divide-y divide-border overflow-hidden rounded-lg border border-border bg-surface">
              {spanModel && (
                <ModelRow role="scrub" desc="detects the PII to protect" model={spanModel} />
              )}
              {relevanceModel !== undefined && (
                <ModelRow
                  role="relevance"
                  desc="keep vs scrub for places & orgs"
                  model={relevanceModel || "off — keyword gate"}
                />
              )}
            </div>
          </section>
        )}

        <section className="mt-7">
          <h2 className="mb-1 text-sm font-medium text-danger">Local data</h2>
          <p className="mb-3 text-xs text-muted">
            Everything Cloak stores stays on this device: your API keys (in the browser), plus your
            saved conversations and learned profile (on disk). You can erase all of it.
          </p>
          <div className="flex items-center justify-between gap-4 rounded-lg border border-danger/30 bg-danger/5 px-4 py-3">
            <div className="text-sm text-muted">
              Deletes saved conversations, the learned profile, and your stored API keys from this
              device. This cannot be undone.
            </div>
            <button
              type="button"
              onClick={() => {
                if (
                  window.confirm(
                    "Clear all local data?\n\nThis permanently deletes your saved conversations, learned profile, and stored API keys from this device.",
                  )
                ) {
                  onWipe()
                }
              }}
              className="shrink-0 rounded-md border border-danger/50 px-3 py-1.5 text-sm text-danger transition-colors hover:bg-danger/10"
            >
              Clear all local data
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}

// Maps an on-device model id to its HuggingFace page (where you can inspect/download it).
const HF_MODEL_URL: Record<string, string> = {
  "praxis/spanfinder-3b": "https://huggingface.co/praxis-nation/spanfinder-3b",
  "praxis/relevance-3b": "https://huggingface.co/praxis-nation/relevance-3b",
}

function ModelRow({ role, desc, model }: { role: string; desc: string; model: string }) {
  const href = HF_MODEL_URL[model]
  return (
    <div className="flex items-center justify-between gap-4 px-3 py-2.5">
      <div>
        <div className="text-sm">{role}</div>
        <div className="text-xs text-muted">{desc}</div>
      </div>
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 font-mono text-xs text-muted underline-offset-2 transition-colors hover:text-accent hover:underline"
        >
          {model}
        </a>
      ) : (
        <code className="shrink-0 font-mono text-xs text-muted">{model}</code>
      )}
    </div>
  )
}

/** A single BYOK provider-key field: masked by default, with a status dot and reveal toggle.
 *  Writes straight through to byok.ts via onChange — there is no separate "save" step. */
function KeyRow({
  title,
  placeholder,
  value,
  onChange,
}: {
  title: string
  placeholder: string
  value: string
  onChange: (key: string) => void
}) {
  const [show, setShow] = useState(false)
  const set = value.trim().length > 0
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-3">
      <div className="mb-1.5 flex items-center gap-2">
        <span
          className={set ? "h-2 w-2 rounded-full bg-protected" : "h-2 w-2 rounded-full bg-border"}
          aria-hidden
        />
        <span className="text-sm">{title}</span>
        <span className="text-xs text-muted">{set ? "key set" : "not set"}</span>
      </div>
      <div className="flex items-center gap-2">
        <input
          type={show ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck={false}
          className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1.5 font-mono text-[13px] text-foreground placeholder:text-muted focus:border-accent focus:outline-none"
        />
        <button
          type="button"
          onClick={() => setShow((s) => !s)}
          aria-label={show ? "Hide key" : "Show key"}
          title={show ? "Hide key" : "Show key"}
          className="shrink-0 rounded-md border border-border p-1.5 text-muted transition-colors hover:text-foreground"
        >
          {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
        </button>
        {set && (
          <button
            type="button"
            onClick={() => onChange("")}
            className="shrink-0 rounded-md border border-border px-2 py-1.5 text-[13px] text-muted transition-colors hover:text-danger"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  )
}

function Row({ title, desc, children }: { title: string; desc: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-border bg-surface px-3 py-3">
      <div>
        <div className="text-sm">{title}</div>
        <div className="text-xs text-muted">{desc}</div>
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

function Stepper({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  const btn =
    "grid h-6 w-6 place-items-center rounded-md border border-border text-muted transition-colors hover:text-foreground"
  return (
    <div className="flex items-center gap-1.5">
      <button type="button" onClick={() => onChange(Math.max(1, value - 1))} className={btn} aria-label="Fewer">
        −
      </button>
      <span className="w-5 text-center text-sm tabular-nums">{value}</span>
      <button type="button" onClick={() => onChange(value + 1)} className={btn} aria-label="More">
        +
      </button>
    </div>
  )
}
