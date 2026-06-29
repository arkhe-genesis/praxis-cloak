import { useEffect, useRef, useState } from "react"
import { KeyRound, Loader2, Save, SaveOff } from "lucide-react"
import * as Tooltip from "@radix-ui/react-tooltip"
import * as api from "@/lib/api"
import { clearAllKeys, getByokKeys, setAnthropicKey, setOpenaiKey, type ByokKeys } from "@/lib/byok"
import type {
  Change,
  HealthInfo,
  ModelCatalog,
  ProfileOverview,
  ScrubResult,
  SendMode,
  SessionMeta,
} from "@/lib/api"
import type { ChatMsg } from "@/lib/types"
import { ChatMessage } from "@/components/ChatMessage"
import { HowItWorks } from "@/components/HowItWorks"
import { ComposerArea } from "@/components/ComposerArea"
import { ProfileView } from "@/components/ProfileView"
import { SettingsView } from "@/components/SettingsView"
import { Sidebar, type View } from "@/components/Sidebar"

const uid = () => crypto.randomUUID()
const SIDEBAR_KEY = "gk_sidebar_collapsed"

export default function App() {
  // Open-source, no-auth build: render the workspace directly. Cloud chat is BYOK — the user
  // supplies their own provider key in Settings; there is no sign-in gate.
  return <Workspace />
}

function Workspace() {
  const [byok, setByok] = useState<ByokKeys>(() => getByokKeys())
  const hasAnyKey = Boolean(byok.anthropic || byok.openai)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [draft, setDraft] = useState("")
  const [preview, setPreview] = useState<ScrubResult | null>(null)
  const [autoApprove, setAutoApprove] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null)
  const [connStatus, setConnStatus] = useState<"connecting" | "ready" | "failed">("connecting")

  const [sessions, setSessions] = useState<SessionMeta[]>([])
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [currentSave, setCurrentSave] = useState(true)
  const [defaultSave, setDefaultSave] = useState(true)
  const [learnFromChats, setLearnFromChats] = useState(true)
  const [promoteThreshold, setPromoteThreshold] = useState(2)
  const [view, setView] = useState<View>("chat")
  const [showHelp, setShowHelp] = useState(false)
  const [profile, setProfile] = useState<ProfileOverview | null>(null)
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null)
  const [model, setModel] = useState<string>("") // the CURRENT chat's model (per-session)
  const [effort, setEffort] = useState<string>("")
  const [defaultModel, setDefaultModel] = useState<string>("") // setting: model new chats start on
  const [defaultEffort, setDefaultEffort] = useState<string>("")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === "1",
  )
  const scrollRef = useRef<HTMLDivElement>(null)

  async function init() {
    setConnStatus("connecting")
    setError(null)
    // The backend is spawned alongside the app and may take a few seconds to listen.
    // Poll /health with backoff over a generous window — staying in "connecting" — so a
    // slow start just shows the connecting state longer rather than a hard failure.
    let info: HealthInfo | null = null
    const deadline = Date.now() + 30000
    let attempt = 0
    while (!info && Date.now() < deadline) {
      try {
        info = await api.health()
      } catch {
        await new Promise((r) => setTimeout(r, Math.min(300 + attempt * 200, 1500)))
        attempt++
      }
    }
    if (!info) {
      setError("Can't reach the local engine.")
      setConnStatus("failed")
      return
    }
    setHealthInfo(info)
    try {
      setCatalog(await api.getModels())
    } catch {
      /* ignore */
    }
    try {
      const s = await api.getSettings()
      setDefaultSave(s.default_save_history)
      setLearnFromChats(s.learn_from_chats)
      setPromoteThreshold(s.promote_threshold)
      setDefaultModel(s.default_model)
      setDefaultEffort(s.default_effort)
    } catch {
      /* ignore */
    }
    try {
      setSessions(await api.listSessions())
    } catch {
      /* ignore */
    }
    try {
      const s = await api.createSession()
      setCurrentId(s.id)
      setCurrentSave(s.save_enabled)
      setModel(s.model)
      setEffort(s.effort)
      setConnStatus("ready")
      void refreshProfile()
    } catch (e) {
      setError("Couldn't start a session: " + (e as Error).message)
      setConnStatus("failed")
    }
  }

  async function refreshProfile() {
    try {
      setProfile(await api.getProfile())
    } catch {
      /* ignore */
    }
  }

  async function promoteEntity(id: number) {
    await api.promoteProfileEntity(id).catch(() => {})
    await refreshProfile()
  }

  async function rejectEntity(id: number) {
    await api.rejectProfileEntity(id).catch(() => {})
    await refreshProfile()
  }

  async function addEntity(surface: string, category: string) {
    await api.addProfileEntity(surface, category).catch(() => {})
    await refreshProfile()
  }

  function navigate(v: View) {
    setView(v)
    if (v === "profile") void refreshProfile() // open with fresh pending/active
  }

  useEffect(() => {
    void init()
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  async function refreshSessions() {
    try {
      setSessions(await api.listSessions())
    } catch {
      /* ignore */
    }
  }

  async function doSend(mode: SendMode, sentText: string, changes: Change[], original: string) {
    if (!currentId) return
    setPreview(null)
    setDraft("")
    const userId = uid()
    const botId = uid()
    setMessages((m) => [
      ...m,
      {
        id: userId,
        role: "user",
        text: original,
        modelSaw: sentText,
        changes: mode === "original" ? [] : changes,
        mode,
      },
      { id: botId, role: "assistant", text: "", streaming: true },
    ])
    setBusy(true)
    try {
      await api.sendStream(currentId, { mode, text: mode === "edited" ? sentText : undefined, model, effort }, (ev) => {
        setMessages((m) =>
          m.map((x) => {
            if (ev.type === "sent" && x.id === userId) return { ...x, modelSaw: ev.model_saw }
            if (ev.type === "delta" && x.id === botId) return { ...x, text: ev.answer }
            if (ev.type === "done" && x.id === botId)
              return {
                ...x,
                text: ev.answer,
                modelSaw: ev.answer_model_saw,
                rehydrations: ev.rehydrations,
                streaming: false,
              }
            if (ev.type === "error" && x.id === botId) return { ...x, streaming: false, error: ev.error }
            return x
          }),
        )
      })
      await refreshSessions() // pick up the newly-persisted session + title/recency
      await refreshProfile() // the send may have advanced a candidate toward promotion (dot)
    } catch (e) {
      setMessages((m) =>
        m.map((x) => (x.id === botId ? { ...x, streaming: false, error: (e as Error).message } : x)),
      )
    } finally {
      setBusy(false)
    }
  }

  async function submitDraft() {
    const text = draft.trim()
    if (!text || busy || !currentId) return
    setError(null)
    setBusy(true)
    try {
      const result = await api.scrub(currentId, text)
      if (result.changes.length === 0 || autoApprove) {
        await doSend("protected", result.protected, result.changes, text)
      } else {
        setPreview(result)
        setBusy(false)
      }
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  function onDraftChange(v: string) {
    setDraft(v)
    if (preview) setPreview(null)
  }

  function approve(mode: SendMode, text: string, always: boolean) {
    if (!preview) return
    if (always) setAutoApprove(true)
    doSend(mode, text, preview.changes, preview.original)
  }

  function sendOriginal() {
    if (!preview) return
    doSend("original", preview.original, [], preview.original)
  }

  async function newConversation() {
    setView("chat")
    try {
      const s = await api.createSession()
      setCurrentId(s.id)
      setCurrentSave(s.save_enabled)
      setModel(s.model)
      setEffort(s.effort)
    } catch (e) {
      setError((e as Error).message)
    }
    setMessages([])
    setDraft("")
    setPreview(null)
    setError(null)
  }

  async function selectSession(id: string) {
    setView("chat")
    if (id === currentId) return
    try {
      const d = await api.getSession(id)
      setCurrentId(d.id)
      setCurrentSave(d.save_enabled)
      setModel(d.model)
      setEffort(d.effort)
      setMessages(
        d.messages.map((m) => ({
          id: uid(),
          role: m.role,
          text: m.text,
          modelSaw: m.modelSaw,
          changes: m.changes,
          rehydrations: m.rehydrations,
          mode: m.mode,
        })),
      )
      setDraft("")
      setPreview(null)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function removeSession(id: string) {
    await api.deleteSession(id).catch(() => {})
    await refreshSessions()
    if (id === currentId) await newConversation()
  }

  async function renameSession(id: string, title: string) {
    setSessions((ss) => ss.map((s) => (s.id === id ? { ...s, title } : s))) // optimistic
    await api.patchSession(id, { title }).catch(() => {})
    await refreshSessions()
  }

  function toggleSidebar() {
    setSidebarCollapsed((c) => {
      const next = !c
      localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0")
      return next
    })
  }

  function selectModel(modelId: string, effortValue: string) {
    setModel(modelId)
    setEffort(effortValue)
    // remember this chat's choice (per-session memory; persists if the chat is saved)
    if (currentId) void api.patchSession(currentId, { model: modelId, effort: effortValue }).catch(() => {})
    // the user's pick also becomes the default for new chats (sticky last choice), so we never
    // fall back to a preset model they didn't choose
    setDefaultModel(modelId)
    setDefaultEffort(effortValue)
    void api.putSettings({ default_model: modelId, default_effort: effortValue }).catch(() => {})
  }

  async function changeDefaultModel(modelId: string, effortValue: string) {
    setDefaultModel(modelId)
    setDefaultEffort(effortValue)
    await api.putSettings({ default_model: modelId, default_effort: effortValue }).catch(() => {})
  }

  async function toggleSaveCurrent(v: boolean) {
    if (!currentId) return
    setCurrentSave(v)
    await api.patchSession(currentId, { save_enabled: v }).catch(() => {})
    await refreshSessions()
  }

  async function toggleDefaultSave(v: boolean) {
    setDefaultSave(v)
    await api.putSettings({ default_save_history: v }).catch(() => {})
  }

  async function toggleLearn(v: boolean) {
    setLearnFromChats(v)
    await api.putSettings({ learn_from_chats: v }).catch(() => {})
  }

  async function changeThreshold(n: number) {
    setPromoteThreshold(n)
    await api.putSettings({ promote_threshold: n }).catch(() => {})
    await refreshProfile() // the pending/watching split depends on the threshold
  }

  async function editEntity(id: number, patch: { surface?: string; category?: string; variants?: string[] }) {
    await api.updateProfileEntity(id, patch).catch(() => {})
    await refreshProfile()
  }

  // BYOK: provider keys live in localStorage only and ride along with scrub/send to the
  // local backend. Mirror them into state so the empty-state/banner stays in sync.
  function changeAnthropicKey(key: string) {
    setAnthropicKey(key)
    setByok(getByokKeys())
  }
  function changeOpenaiKey(key: string) {
    setOpenaiKey(key)
    setByok(getByokKeys())
  }

  // "Clear all local data": erase conversations + learned profile (server) and API keys +
  // prefs (browser), then reload so every bit of in-memory state resets to a clean slate.
  async function clearAllLocalData() {
    await api.wipeLocalData().catch(() => {})
    clearAllKeys()
    localStorage.removeItem(SIDEBAR_KEY)
    window.location.reload()
  }

  return (
    <Tooltip.Provider delayDuration={150} skipDelayDuration={400}>
      <div className="flex h-screen">
        <Sidebar
          sessions={sessions}
          currentId={currentId}
          view={view}
          collapsed={sidebarCollapsed}
          onToggleCollapse={toggleSidebar}
          onSelect={selectSession}
          onNew={newConversation}
          onDelete={removeSession}
          onRename={renameSession}
          onNavigate={navigate}
          onShowHelp={() => setShowHelp(true)}
          pendingCount={profile?.pending_count ?? 0}
        />

        <HowItWorks open={showHelp} onClose={() => setShowHelp(false)} />

        <div className="flex min-w-0 flex-1 flex-col">
          {view === "profile" ? (
            <ProfileView
              overview={profile}
              onPromote={promoteEntity}
              onReject={rejectEntity}
              onAdd={addEntity}
              onEdit={editEntity}
            />
          ) : view === "settings" ? (
            <SettingsView
              defaultSave={defaultSave}
              onToggleDefaultSave={toggleDefaultSave}
              learnFromChats={learnFromChats}
              onToggleLearn={toggleLearn}
              promoteThreshold={promoteThreshold}
              onChangeThreshold={changeThreshold}
              catalog={catalog}
              defaultModel={defaultModel}
              defaultEffort={defaultEffort}
              onSelectDefaultModel={changeDefaultModel}
              byok={byok}
              onChangeAnthropicKey={changeAnthropicKey}
              onChangeOpenaiKey={changeOpenaiKey}
              spanModel={healthInfo?.local_model}
              relevanceModel={healthInfo?.relevance_model}
              onWipe={clearAllLocalData}
            />
          ) : (
            <div className="relative flex-1 overflow-hidden">
              {/* Floating save-history toggle — no header, no separator line. */}
              <div className="absolute right-3 top-3 z-10">
                <SaveHistoryToggle enabled={currentSave} onToggle={toggleSaveCurrent} />
              </div>

              <div ref={scrollRef} className="h-full overflow-y-auto">
                <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 pb-44 pt-12">
                  {connStatus === "ready" && !hasAnyKey && messages.length > 0 && (
                    <NoKeyBanner onOpenSettings={() => navigate("settings")} />
                  )}
                  {connStatus === "connecting" ? (
                    <Connecting />
                  ) : connStatus === "failed" ? (
                    <ConnFailed message={error} onRetry={() => void init()} />
                  ) : messages.length === 0 ? (
                    <EmptyState needsKey={!hasAnyKey} onOpenSettings={() => navigate("settings")} />
                  ) : (
                    messages.map((m) => <ChatMessage key={m.id} msg={m} />)
                  )}
                  {connStatus === "ready" && error && (
                    <div className="mx-auto rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
                      {error}
                    </div>
                  )}
                </div>
              </div>

              {/* Composer floats over the content, which scrolls behind a soft fade. */}
              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col">
                <div className="h-10 bg-gradient-to-t from-background to-transparent" />
                <div className="pointer-events-auto bg-background">
                  <ComposerArea
                    draft={draft}
                    onDraftChange={onDraftChange}
                    onSubmitDraft={submitDraft}
                    preview={preview}
                    busy={busy || connStatus !== "ready"}
                    onApprove={approve}
                    onSendOriginal={sendOriginal}
                    onCancelReview={() => setPreview(null)}
                    catalog={catalog}
                    model={model}
                    effort={effort}
                    onSelectModel={selectModel}
                    byok={byok}
                    autoApprove={autoApprove}
                    onClearAutoApprove={() => setAutoApprove(false)}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Tooltip.Provider>
  )
}

/** The only chrome left at the top of the chat: a quiet, floating toggle for whether this
 *  conversation is being saved. Save = persisted (scrubbed) locally; SaveOff = ephemeral. */
function SaveHistoryToggle({ enabled, onToggle }: { enabled: boolean; onToggle: (v: boolean) => void }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button
          type="button"
          onClick={() => onToggle(!enabled)}
          aria-label={enabled ? "Turn off save history" : "Turn on save history"}
          aria-pressed={enabled}
          className="rounded-md p-1.5 text-muted backdrop-blur transition-colors hover:bg-surface hover:text-foreground"
        >
          {enabled ? <Save className="h-4 w-4 text-protected" /> : <SaveOff className="h-4 w-4" />}
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={5}
          className="z-50 max-w-56 select-none rounded-md border border-border bg-surface px-2 py-1 text-xs text-foreground shadow-lg"
        >
          {enabled
            ? "Saving history (scrubbed, on this device) — click to turn off"
            : "Not saving history — click to turn on"}
          <Tooltip.Arrow className="fill-[var(--surface)]" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}

function Connecting() {
  return (
    <div className="mx-auto mt-20 flex max-w-md flex-col items-center gap-3 text-center">
      <Loader2 className="h-5 w-5 animate-spin text-muted" />
      <p className="text-sm text-muted">Connecting to the local engine…</p>
    </div>
  )
}

function ConnFailed({ message, onRetry }: { message: string | null; onRetry: () => void }) {
  return (
    <div className="mx-auto mt-20 flex max-w-md flex-col items-center gap-3 text-center">
      <p className="text-sm text-muted">{message ?? "Can't reach the local engine."}</p>
      <button
        type="button"
        onClick={onRetry}
        className="rounded-md border border-border px-3 py-1.5 text-sm text-foreground transition-colors hover:border-accent/60"
      >
        Retry
      </button>
    </div>
  )
}

function EmptyState({ needsKey, onOpenSettings }: { needsKey: boolean; onOpenSettings: () => void }) {
  return (
    <div className="mx-auto mt-16 max-w-md text-center">
      <h1 className="mb-2 text-lg font-semibold">Chat privately</h1>
      <p className="text-sm leading-relaxed text-muted">
        Type a message to start — your PII is swapped out on-device before anything is sent.
      </p>
      {needsKey && (
        <div className="mx-auto mt-6 flex max-w-sm flex-col items-center gap-3 rounded-xl border border-accent/40 bg-accent/5 px-5 py-4">
          <KeyRound className="h-5 w-5 text-accent" />
          <p className="text-sm text-foreground">
            Add your own Anthropic or OpenAI API key to start chatting.
          </p>
          <p className="text-xs text-muted">Keys stay on this device and are only sent to your local engine.</p>
          <button
            type="button"
            onClick={onOpenSettings}
            className="mt-1 rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-accent-foreground transition hover:brightness-110"
          >
            Add a key in Settings
          </button>
        </div>
      )}
    </div>
  )
}

/** A slim, persistent prompt when no provider key is set but a conversation is in progress —
 *  so a send that's about to fail for lack of a key never fails silently. */
function NoKeyBanner({ onOpenSettings }: { onOpenSettings: () => void }) {
  return (
    <div className="mx-auto flex w-full max-w-2xl items-center gap-3 rounded-lg border border-accent/40 bg-accent/5 px-3 py-2">
      <KeyRound className="h-4 w-4 shrink-0 text-accent" />
      <span className="flex-1 text-sm text-foreground">
        No provider key set — add your Anthropic or OpenAI key to send messages.
      </span>
      <button
        type="button"
        onClick={onOpenSettings}
        className="shrink-0 rounded-md bg-accent px-2.5 py-1 text-[13px] font-medium text-accent-foreground transition hover:brightness-110"
      >
        Open Settings
      </button>
    </div>
  )
}
