import type { ReactNode } from "react"
import * as Dialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"

/** "How it works" modal, opened from the help (?) button in the sidebar. */
export function HowItWorks({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[92vw] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-2xl border border-border bg-surface p-6 shadow-2xl focus:outline-none">
          <div className="mb-4 flex items-start justify-between gap-4">
            <Dialog.Title className="text-lg font-semibold">How Cloak works</Dialog.Title>
            <Dialog.Close
              className="rounded-md p-1 text-muted transition-colors hover:text-foreground"
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>

          <Dialog.Description className="text-sm leading-relaxed text-muted">
            Type a message. Before it leaves your device, Cloak replaces names, places, and other
            identifying details with realistic stand-ins, and shows you exactly what the model will
            see. The reply is mapped back to your real world locally.
          </Dialog.Description>

          <div className="mt-5 space-y-3 text-sm">
            <Step n={1} title="Detect">
              A fast regex pass catches hard identifiers (emails, phone numbers, API keys), and a
              small on-device model finds softer PII — names, employers, locations.
            </Step>
            <Step n={2} title="Decide (relevance)">
              Not everything should be hidden. A second on-device model judges whether each detail is{" "}
              <em>load-bearing</em> — needed to answer your question — or incidental. "What's the tax
              rate in Toronto?" keeps "Toronto"; a coworker's name mentioned in passing gets swapped.
            </Step>
            <Step n={3} title="Substitute">
              Incidental details are replaced with realistic fakes — actual stand-in names and cities,
              not <code>[REDACTED]</code> — so the cloud model still sees a coherent, answerable prompt.
            </Step>
            <Step n={4} title="Rehydrate">
              When the reply comes back, the fakes are mapped to your real entities locally, so the
              answer reads as if nothing was ever changed.
            </Step>
          </div>

          <Section title="What counts as PII">
            Names, emails, phone numbers, addresses, employers, locations, account numbers and other
            identifiers. Hard identifiers are caught by deterministic rules; everything else is
            detected by the on-device model. Substitutions stay consistent within a conversation, so
            the same person keeps the same stand-in throughout.
          </Section>

          <Section title="Why relevance matters">
            Blunt redaction breaks answers: scrub the city from a question about local law and the
            model can't help you. The relevance model keeps the details an answer genuinely depends
            on and only swaps the rest — protection without losing usefulness.
          </Section>

          <p className="mt-5 text-xs text-muted">
            This is <strong className="text-foreground">disclosure control</strong>, not a privacy
            guarantee. Everything above runs on your device; only the scrubbed prompt reaches the
            cloud model, using your own API key.
          </p>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/15 text-[11px] font-semibold text-accent">
        {n}
      </span>
      <div>
        <div className="font-medium text-foreground">{title}</div>
        <p className="text-muted">{children}</p>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mt-5">
      <h3 className="mb-1 text-sm font-medium text-foreground">{title}</h3>
      <p className="text-sm leading-relaxed text-muted">{children}</p>
    </div>
  )
}
