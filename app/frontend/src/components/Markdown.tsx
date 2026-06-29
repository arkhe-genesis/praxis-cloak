import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import * as Tooltip from "@radix-ui/react-tooltip"
import type { Mark } from "@/components/HighlightedText"

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

// A self-contained rehype plugin: split text nodes around the marked entities and wrap
// each in a <mark> carrying its provenance tip. Runs on the parsed tree so it never
// touches code/pre or breaks markdown structure. No unist-util-visit dependency — a
// small recursive walk suffices and lets us skip code spans/blocks.
function highlightPlugin(marks: Mark[]) {
  const byNeedle = new Map<string, Mark>()
  for (const m of marks) if (m.needle) byNeedle.set(m.needle, m)
  const needles = Array.from(byNeedle.keys())
  const SKIP = new Set(["code", "pre"])

  return function attacher() {
    return function transform(tree: any) {
      if (needles.length === 0) return
      const re = new RegExp(`(${needles.map(escapeRe).join("|")})`, "g")
      walk(tree, false)

      function walk(node: any, inSkip: boolean) {
        if (!node.children) return
        const out: any[] = []
        for (const child of node.children) {
          if (child.type === "element") {
            walk(child, inSkip || SKIP.has(child.tagName))
            out.push(child)
          } else if (child.type === "text" && !inSkip) {
            for (const part of child.value.split(re)) {
              if (part === "") continue
              const m = byNeedle.get(part)
              out.push(
                m
                  ? {
                      type: "element",
                      tagName: "mark",
                      properties: { dataTip: m.tip, dataProfile: m.profile ? "true" : undefined },
                      children: [{ type: "text", value: part }],
                    }
                  : { type: "text", value: part },
              )
            }
          } else {
            out.push(child)
          }
        }
        node.children = out
      }
    }
  }
}

/** Render an LLM reply as markdown (GFM), styled via the `.gk-markdown` scope. Entities
 *  restored locally are highlighted inline with a hover tooltip showing what the model
 *  actually wrote in their place. */
export function Markdown({ text, marks = [] }: { text: string; marks?: Mark[] }) {
  return (
    <div className="gk-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[highlightPlugin(marks)]}
        components={{
          a: ({ node: _n, ...props }) => <a target="_blank" rel="noreferrer" {...props} />,
          mark: ({ node, children }) => {
            const tip = (node?.properties?.dataTip as string) || ""
            const profile = node?.properties?.dataProfile === "true"
            const cls =
              "cursor-help rounded-sm bg-transparent text-inherit underline decoration-protected/60 decoration-dotted underline-offset-2" +
              (profile ? " ring-1 ring-protected/50" : "")
            if (!tip) return <mark className={cls}>{children}</mark>
            return (
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <mark className={cls}>{children}</mark>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    sideOffset={5}
                    className="z-50 select-none rounded-md border border-border bg-surface px-2 py-1 text-xs text-foreground shadow-lg"
                  >
                    {tip}
                    <Tooltip.Arrow className="fill-[var(--surface)]" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            )
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
