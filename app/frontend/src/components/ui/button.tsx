import * as React from "react"
import { cn } from "@/lib/utils"

type Variant = "primary" | "secondary" | "ghost" | "danger"
type Size = "sm" | "md"

const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-foreground hover:brightness-110",
  secondary: "border border-border bg-surface-2 text-foreground hover:border-accent/60",
  ghost: "bg-transparent text-muted hover:bg-surface-2 hover:text-foreground",
  danger: "border border-danger/40 bg-transparent text-danger hover:bg-danger/10",
}

const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-[13px]",
  md: "h-10 px-4 text-sm",
}

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
}

export function Button({ className, variant = "secondary", size = "md", ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
        "disabled:cursor-not-allowed disabled:opacity-50",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  )
}
