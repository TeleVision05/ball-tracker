import type { ReactNode } from 'react'

interface AppShellProps {
  sidebar: ReactNode
  main: ReactNode
  rightPanel: ReactNode | null
}

export function AppShell({ sidebar, main, rightPanel }: AppShellProps) {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-base-950 text-white/90">
      <aside className="flex w-[240px] shrink-0 flex-col border-r border-white/[0.08] bg-base-900">
        {sidebar}
      </aside>
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden bg-base-950">{main}</main>
      {rightPanel && (
        <aside className="flex w-[300px] shrink-0 flex-col border-l border-white/[0.08] bg-base-900 overflow-y-auto">
          {rightPanel}
        </aside>
      )}
    </div>
  )
}
