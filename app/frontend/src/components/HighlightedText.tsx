import * as Tooltip from "@radix-ui/react-tooltip"
import { cn } from "@/lib/utils"

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export interface Mark {
  needle: string
  tip: string
  profile?: boolean // matched the user's saved profile — gets a subtle distinguishing ring
}

/** Render `text`, wrapping each marked substring in a highlight whose hover tooltip shows
 *  the substitution mapping (e.g. "Anna → Mira · Name"). */
export function HighlightedText({
  text,
  marks,
  markClassName,
}: {
  text: string
  marks: Mark[]
  markClassName?: string
}) {
  const byNeedle = new Map<string, Mark>()
  for (const m of marks) if (m.needle) byNeedle.set(m.needle, m)
  const needles = Array.from(byNeedle.keys())
  if (needles.length === 0) return <>{text}</>

  const re = new RegExp(`(${needles.map(escapeRe).join("|")})`, "g")
  return (
    <>
      {text.split(re).map((part, i) => {
        const m = byNeedle.get(part)
        return m ? (
          <Tooltip.Root key={i}>
            <Tooltip.Trigger asChild>
              <mark
                className={cn(
                  "bg-transparent text-inherit",
                  markClassName,
                  m.profile && "rounded-sm ring-1 ring-protected/50",
                )}
              >
                {part}
              </mark>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content
                sideOffset={5}
                className="z-50 select-none rounded-md border border-border bg-surface px-2 py-1 text-xs text-foreground shadow-lg"
              >
                {m.tip}
                <Tooltip.Arrow className="fill-[var(--surface)]" />
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        ) : (
          <span key={i}>{part}</span>
        )
      })}
    </>
  )
}
