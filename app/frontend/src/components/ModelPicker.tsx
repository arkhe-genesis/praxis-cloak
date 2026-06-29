import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import { Check, ChevronDown, ChevronRight } from "lucide-react"
import type { ModelCatalog, ModelInfo } from "@/lib/api"
import type { ByokKeys } from "@/lib/byok"
import { cn } from "@/lib/utils"

/** In BYOK mode a provider counts as available if the user has set its key locally, even
 *  when the backend has no env key of its own. */
function hasByokFor(provider: string, byok?: ByokKeys): boolean {
  if (!byok) return false
  if (provider === "anthropic") return Boolean(byok.anthropic)
  if (provider === "openai") return Boolean(byok.openai)
  return false
}

/** The composer's cloud-model picker (left of the send button, à la ChatGPT/Claude).
 *  Models are grouped by provider and indented under the provider label. Each model with a
 *  reasoning control (effort / intelligence / thinking) opens a nested flyout of its values;
 *  picking a value selects that model + value together. Opens upward (bottom of window). */
export function ModelPicker({
  catalog,
  model,
  effort,
  onSelect,
  byok,
  disabled,
  side = "top",
}: {
  catalog: ModelCatalog | null
  model: string
  effort: string
  onSelect: (model: string, effort: string) => void
  byok?: ByokKeys
  disabled?: boolean
  side?: "top" | "bottom"
}) {
  if (!catalog) return null
  const available = (provider: string, backend: boolean) => backend || hasByokFor(provider, byok)
  const current = catalog.models.find((m) => m.id === model)
  const currentUnavailable = current ? !available(current.provider, current.available) : false
  // Show a neutral "Select model" prompt when there's no usable current model (no key for its
  // provider, or none chosen yet) rather than surfacing an unusable preset.
  const showPrompt = !current || currentUnavailable
  const effortLabel = current?.reasoning?.options.find((o) => o.value === effort)?.label

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild disabled={disabled}>
        <button
          type="button"
          className={cn(
            "flex items-center gap-1 rounded-md px-2 py-1 text-[13px] transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:opacity-50",
            "text-muted hover:text-foreground",
          )}
          title={
            currentUnavailable
              ? "Add a provider key in Settings, then choose a model"
              : "Choose model"
          }
        >
          <span className="max-w-44 truncate">
            {/* Don't surface a preset model the user can't use (no key for its provider) —
                prompt them to pick instead. Their first pick becomes the default. */}
            {showPrompt ? "Select model" : current?.label ?? model}
            {!showPrompt && effortLabel && (
              <span className="text-muted/80"> · {effortLabel}</span>
            )}
          </span>
          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          side={side}
          align="start"
          sideOffset={6}
          className="z-50 min-w-60 rounded-lg border border-border bg-surface p-1 shadow-xl"
        >
          {catalog.providers.map((p) => {
            const models = catalog.models.filter((m) => m.provider === p.id)
            if (models.length === 0) return null
            const providerAvailable = available(p.id, p.available)
            return (
              <DropdownMenu.Group key={p.id}>
                <DropdownMenu.Label className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-muted">
                  {p.label}
                  {!providerAvailable && (
                    <span className="ml-1 normal-case tracking-normal text-danger/80">
                      · add key in Settings
                    </span>
                  )}
                </DropdownMenu.Label>
                {models.map((m) => (
                  <ModelRow
                    key={m.id}
                    m={m}
                    available={available(m.provider, m.available)}
                    selectedModel={model}
                    selectedEffort={effort}
                    onSelect={onSelect}
                  />
                ))}
              </DropdownMenu.Group>
            )
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

function ModelRow({
  m,
  available,
  selectedModel,
  selectedEffort,
  onSelect,
}: {
  m: ModelInfo
  available: boolean
  selectedModel: string
  selectedEffort: string
  onSelect: (model: string, effort: string) => void
}) {
  const isSelected = m.id === selectedModel
  // models are indented one notch past the provider label (pl-4 vs the label's px-2)
  const rowBase =
    "flex items-center gap-2 rounded-md py-1.5 pl-4 pr-2 text-sm outline-none data-[disabled]:cursor-not-allowed data-[disabled]:opacity-40"

  // No reasoning control: a plain selectable item (defaults to empty effort).
  if (!m.reasoning) {
    return (
      <DropdownMenu.Item
        disabled={!available}
        onSelect={() => onSelect(m.id, "")}
        className={cn(rowBase, "cursor-pointer data-[highlighted]:bg-surface-2")}
      >
        <span className={cn("flex-1", isSelected && "text-foreground")}>{m.label}</span>
        {isSelected && <Check className="h-3.5 w-3.5 text-accent" />}
      </DropdownMenu.Item>
    )
  }

  const effortLabel = m.reasoning.options.find((o) => o.value === selectedEffort)?.label
  return (
    <DropdownMenu.Sub>
      <DropdownMenu.SubTrigger
        disabled={!available}
        className={cn(rowBase, "data-[highlighted]:bg-surface-2 data-[state=open]:bg-surface-2")}
      >
        <span className={cn("flex-1", isSelected ? "text-foreground" : "text-muted")}>{m.label}</span>
        {isSelected && effortLabel && <span className="text-xs text-muted">{effortLabel}</span>}
        {isSelected && <Check className="h-3.5 w-3.5 text-accent" />}
        <ChevronRight className="h-3.5 w-3.5 text-muted" />
      </DropdownMenu.SubTrigger>
      <DropdownMenu.Portal>
        <DropdownMenu.SubContent
          sideOffset={4}
          alignOffset={-4}
          className="z-50 min-w-44 rounded-lg border border-border bg-surface p-1 shadow-xl"
        >
          <DropdownMenu.Label className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-muted">
            {m.reasoning.label}
          </DropdownMenu.Label>
          {m.reasoning.options.map((o) => {
            const on = isSelected && o.value === selectedEffort
            return (
              <DropdownMenu.Item
                key={o.value}
                onSelect={() => onSelect(m.id, o.value)}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none data-[highlighted]:bg-surface-2"
              >
                <span className="flex-1">{o.label}</span>
                {on && <Check className="h-3.5 w-3.5 text-accent" />}
              </DropdownMenu.Item>
            )
          })}
        </DropdownMenu.SubContent>
      </DropdownMenu.Portal>
    </DropdownMenu.Sub>
  )
}
