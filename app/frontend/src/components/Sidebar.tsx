import { useEffect, useRef, useState } from "react"
import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import {
  HelpCircle,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Settings,
  SquarePen,
  Trash2,
  UserRound,
} from "lucide-react"
import type { SessionMeta } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export type View = "chat" | "profile" | "settings"

export function Sidebar({
  sessions,
  currentId,
  view,
  collapsed,
  onToggleCollapse,
  onSelect,
  onNew,
  onDelete,
  onRename,
  onNavigate,
  onShowHelp,
  pendingCount,
}: {
  sessions: SessionMeta[]
  currentId: string | null
  view: View
  collapsed: boolean
  onToggleCollapse: () => void
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
  onNavigate: (view: View) => void
  onShowHelp: () => void
  pendingCount: number
}) {
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState("")
  const renameRef = useRef<HTMLInputElement>(null)

  // Steal focus into the rename input once the menu's focus-return has settled.
  useEffect(() => {
    if (!renamingId) return
    const t = setTimeout(() => renameRef.current?.select(), 10)
    return () => clearTimeout(t)
  }, [renamingId])

  function startRename(s: SessionMeta) {
    setRenameValue(s.title)
    setRenamingId(s.id)
  }
  function commitRename(id: string) {
    const v = renameValue.trim()
    if (v) onRename(id, v)
    setRenamingId(null)
  }

  if (collapsed) {
    return (
      <aside className="flex w-14 shrink-0 flex-col items-center border-r border-border bg-background py-3">
        <RailButton label="Expand sidebar" onClick={onToggleCollapse}>
          <PanelLeftOpen className="h-5 w-5" />
        </RailButton>
        <RailButton label="New chat" onClick={onNew}>
          <SquarePen className="h-5 w-5" />
        </RailButton>
        <div className="flex-1" />
        <RailButton label="Profile" active={view === "profile"} onClick={() => onNavigate("profile")}>
          <span className="relative">
            <UserRound className="h-5 w-5" />
            {pendingCount > 0 && (
              <span className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-accent" />
            )}
          </span>
        </RailButton>
        <RailButton label="Settings" active={view === "settings"} onClick={() => onNavigate("settings")}>
          <Settings className="h-5 w-5" />
        </RailButton>
        <RailButton label="How it works" onClick={onShowHelp}>
          <HelpCircle className="h-5 w-5" />
        </RailButton>
      </aside>
    )
  }

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-background">
      <div className="flex items-center justify-between px-4 py-3">
        <span className="flex items-center gap-2 font-semibold tracking-tight">
          <img src="/praxis-wordmark-white.png" alt="Praxis" className="h-4 w-auto" />
          <span className="text-accent">Cloak</span>
        </span>
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Collapse sidebar"
          title="Collapse sidebar"
          className="text-muted transition-colors hover:text-foreground"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
      </div>
      <div className="px-3 pb-2">
        <Button variant="secondary" size="sm" className="w-full" onClick={onNew}>
          <SquarePen className="mr-1.5 h-3.5 w-3.5" /> New chat
        </Button>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 py-1">
        {sessions.length === 0 ? (
          <p className="px-2 py-2 text-xs text-muted">No saved chats yet.</p>
        ) : (
          sessions.map((s) => {
            const isRenaming = renamingId === s.id
            return (
              <div
                key={s.id}
                onClick={() => !isRenaming && onSelect(s.id)}
                className={cn(
                  "group flex cursor-pointer items-center gap-1 rounded-md px-2 py-1.5 text-sm",
                  view === "chat" && s.id === currentId
                    ? "bg-surface-2 text-foreground"
                    : "text-muted hover:bg-surface hover:text-foreground",
                )}
              >
                {isRenaming ? (
                  <input
                    ref={renameRef}
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    onBlur={() => commitRename(s.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename(s.id)
                      if (e.key === "Escape") setRenamingId(null)
                    }}
                    className="min-w-0 flex-1 rounded border border-accent/60 bg-background px-1 py-0.5 text-sm text-foreground focus:outline-none"
                  />
                ) : (
                  <>
                    <span className="flex-1 truncate">{s.title}</span>
                    <SessionMenu onRename={() => startRename(s)} onDelete={() => onDelete(s.id)} />
                  </>
                )}
              </div>
            )
          })
        )}
      </nav>
      <div className="flex flex-col gap-0.5 border-t border-border px-2 py-2">
        <NavItem
          icon={<UserRound className="h-4 w-4" />}
          label="Profile"
          active={view === "profile"}
          badge={pendingCount}
          onClick={() => onNavigate("profile")}
        />
        <NavItem
          icon={<Settings className="h-4 w-4" />}
          label="Settings"
          active={view === "settings"}
          onClick={() => onNavigate("settings")}
        />
        <NavItem
          icon={<HelpCircle className="h-4 w-4" />}
          label="How it works"
          active={false}
          onClick={onShowHelp}
        />
      </div>
    </aside>
  )
}

function SessionMenu({ onRename, onDelete }: { onRename: () => void; onDelete: () => void }) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          aria-label="Chat options"
          onClick={(e) => e.stopPropagation()}
          className="shrink-0 rounded text-muted opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100 data-[state=open]:opacity-100"
        >
          <MoreHorizontal className="h-4 w-4" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          side="bottom"
          align="end"
          sideOffset={4}
          onClick={(e) => e.stopPropagation()}
          className="z-50 min-w-32 rounded-lg border border-border bg-surface p-1 shadow-xl"
        >
          <DropdownMenu.Item
            onSelect={onRename}
            className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none data-[highlighted]:bg-surface-2"
          >
            <Pencil className="h-3.5 w-3.5" /> Rename
          </DropdownMenu.Item>
          <DropdownMenu.Item
            onSelect={onDelete}
            className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm text-danger outline-none data-[highlighted]:bg-danger/10"
          >
            <Trash2 className="h-3.5 w-3.5" /> Delete
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

function RailButton({
  children,
  label,
  active,
  onClick,
}: {
  children: React.ReactNode
  label: string
  active?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "mb-1 grid h-9 w-9 place-items-center rounded-md transition-colors",
        active ? "bg-surface-2 text-foreground" : "text-muted hover:bg-surface hover:text-foreground",
      )}
    >
      {children}
    </button>
  )
}

function NavItem({
  icon,
  label,
  active,
  badge,
  onClick,
}: {
  icon: React.ReactNode
  label: string
  active: boolean
  badge?: number
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
        active ? "bg-surface-2 text-foreground" : "text-muted hover:bg-surface hover:text-foreground",
      )}
    >
      {icon}
      <span className="flex-1 text-left">{label}</span>
      {badge != null && badge > 0 && (
        <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-accent px-1 text-[10px] font-medium text-accent-foreground">
          {badge}
        </span>
      )}
    </button>
  )
}
