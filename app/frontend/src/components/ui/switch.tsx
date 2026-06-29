import { cn } from "@/lib/utils"

export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label?: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="inline-flex items-center gap-2 text-xs text-muted transition-colors hover:text-foreground"
    >
      <span
        className={cn(
          "relative h-4 w-7 shrink-0 rounded-full transition-colors",
          checked ? "bg-protected/70" : "bg-border",
        )}
      >
        <span
          className={cn(
            "absolute left-0.5 top-0.5 h-3 w-3 rounded-full bg-foreground transition-transform",
            checked ? "translate-x-3" : "translate-x-0",
          )}
        />
      </span>
      {label}
    </button>
  )
}
