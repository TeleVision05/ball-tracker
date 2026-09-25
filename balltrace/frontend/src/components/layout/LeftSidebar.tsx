import { Plus, Settings, Radar, Loader2, AlertCircle } from 'lucide-react'
import type { AnalysisSummary } from '../../types'
import { STATUS_DOT, STATUS_LABEL, formatRelativeDate } from '../../lib/status'

interface LeftSidebarProps {
  analyses: AnalysisSummary[]
  analysesLoading: boolean
  analysesError: string | null
  selectedId: string | null
  activeView: 'analysis' | 'settings'
  onSelectAnalysis: (id: string) => void
  onNewAnalysis: () => void
  onOpenSettings: () => void
}

export function LeftSidebar({
  analyses,
  analysesLoading,
  analysesError,
  selectedId,
  activeView,
  onSelectAnalysis,
  onNewAnalysis,
  onOpenSettings,
}: LeftSidebarProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-5 pt-6 pb-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-neon-cyan/10 text-neon-cyan shadow-glow-sm">
          <Radar size={20} strokeWidth={2.25} />
        </div>
        <div className="flex flex-col leading-none">
          <span className="font-display text-[17px] font-bold tracking-tight text-white">
            BallTrace
          </span>
          <span className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.14em] text-white/35">
            Ball Analysis
          </span>
        </div>
      </div>

      <div className="px-4">
        <button onClick={onNewAnalysis} className="btn-primary w-full shadow-glow-sm">
          <Plus size={16} strokeWidth={2.5} />
          New Analysis
        </button>
      </div>

      <div className="mt-6 flex-1 overflow-y-auto px-3">
        <div className="mb-2 px-2">
          <span className="label-text">Recent Analyses</span>
        </div>

        {analysesLoading && (
          <div className="flex items-center gap-2 px-2 py-3 text-sm text-white/40">
            <Loader2 size={14} className="animate-spin" />
            Loading...
          </div>
        )}

        {analysesError && !analysesLoading && (
          <div className="flex items-start gap-2 rounded-md bg-red-500/10 px-2.5 py-2 text-xs text-red-300">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <span>{analysesError}</span>
          </div>
        )}

        {!analysesLoading && !analysesError && analyses.length === 0 && (
          <div className="px-2 py-3 text-sm text-white/30">No analyses yet.</div>
        )}

        <ul className="flex flex-col gap-1">
          {analyses.map((a) => {
            const isActive = activeView === 'analysis' && a.id === selectedId
            return (
              <li key={a.id}>
                <button
                  onClick={() => onSelectAnalysis(a.id)}
                  className={`group flex w-full flex-col gap-1 rounded-md px-2.5 py-2 text-left transition-colors ${
                    isActive
                      ? 'bg-white/[0.07] ring-1 ring-inset ring-neon-cyan/30'
                      : 'hover:bg-white/[0.04]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className={`truncate text-sm font-medium ${
                        isActive ? 'text-white' : 'text-white/75'
                      }`}
                    >
                      {a.name}
                    </span>
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${STATUS_DOT[a.status]}`} />
                  </div>
                  <div className="flex items-center justify-between gap-2 text-[11px] text-white/35">
                    <span>{STATUS_LABEL[a.status]}</span>
                    <span>{formatRelativeDate(a.created_at)}</span>
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      </div>

      <div className="border-t border-white/[0.08] px-3 py-3">
        <button
          onClick={onOpenSettings}
          className={`flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors ${
            activeView === 'settings'
              ? 'bg-white/[0.07] text-white'
              : 'text-white/60 hover:bg-white/[0.04] hover:text-white'
          }`}
        >
          <Settings size={16} />
          Settings
        </button>
      </div>
    </div>
  )
}
