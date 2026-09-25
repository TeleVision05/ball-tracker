import { useState } from 'react'
import { ChevronDown, ChevronUp, Loader2, Radar, XCircle } from 'lucide-react'
import type { ProgressEvent } from '../../types'

interface ProgressPanelProps {
  progress: ProgressEvent | null
  connected: boolean
  streamError: string | null
  onCancel: () => void
  cancelling: boolean
}

function formatSeconds(seconds: number | null): string {
  if (seconds == null || Number.isNaN(seconds)) return '--'
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (mins === 0) return `${secs}s`
  return `${mins}m ${secs}s`
}

export function ProgressPanel({
  progress,
  connected,
  streamError,
  onCancel,
  cancelling,
}: ProgressPanelProps) {
  const [logOpen, setLogOpen] = useState(false)

  const totalWindows = progress?.total_windows ?? 0
  const currentWindow = progress?.current_window ?? 0
  const pct = totalWindows > 0 ? Math.min(100, Math.round((currentWindow / totalWindows) * 100)) : 0

  return (
    <div className="mx-auto w-full max-w-xl">
      <div className="glass-panel rounded-xl p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-neon-cyan/10 text-neon-cyan">
            <Radar size={20} className="animate-pulse-glow" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-white">
              {progress?.message ?? 'Starting analysis…'}
            </p>
            <p className="mt-0.5 flex items-center gap-1.5 text-xs text-white/40">
              {connected ? (
                <>
                  <span className="h-1.5 w-1.5 rounded-full bg-neon-green" /> Live
                </>
              ) : (
                <>
                  <Loader2 size={11} className="animate-spin" /> Connecting…
                </>
              )}
            </p>
          </div>
        </div>

        <div className="mt-5">
          <div className="h-2 w-full overflow-hidden rounded-full bg-white/[0.06]">
            <div
              className="h-full rounded-full bg-neon-cyan shadow-glow transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="mt-2 flex justify-between text-xs text-white/40">
            <span>
              Window {currentWindow} of {totalWindows || '?'}
            </span>
            <span>{pct}%</span>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-3 gap-3">
          <div className="rounded-lg bg-white/[0.03] px-3 py-2.5">
            <p className="label-text">Detections</p>
            <p className="mt-1 font-mono text-lg font-semibold text-white">
              {progress?.detections_count ?? 0}
            </p>
          </div>
          <div className="rounded-lg bg-white/[0.03] px-3 py-2.5">
            <p className="label-text">Elapsed</p>
            <p className="mt-1 font-mono text-lg font-semibold text-white">
              {formatSeconds(progress?.elapsed_seconds ?? 0)}
            </p>
          </div>
          <div className="rounded-lg bg-white/[0.03] px-3 py-2.5">
            <p className="label-text">ETA</p>
            <p className="mt-1 font-mono text-lg font-semibold text-white">
              {formatSeconds(progress?.eta_seconds ?? null)}
            </p>
          </div>
        </div>

        {streamError && (
          <p className="mt-4 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-300">
            {streamError}
          </p>
        )}

        {progress && progress.log.length > 0 && (
          <div className="mt-4">
            <button
              onClick={() => setLogOpen((v) => !v)}
              className="flex items-center gap-1.5 text-xs font-medium text-white/45 hover:text-white/70"
            >
              {logOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              Activity log ({progress.log.length})
            </button>
            {logOpen && (
              <div className="mt-2 max-h-40 overflow-y-auto rounded-md bg-black/30 p-2.5 font-mono text-[11px] leading-relaxed text-white/50">
                {progress.log.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}
          </div>
        )}

        <button
          onClick={onCancel}
          disabled={cancelling}
          className="btn-secondary mt-5 w-full text-red-300 hover:bg-red-500/10 hover:text-red-300"
        >
          <XCircle size={15} />
          {cancelling ? 'Cancelling…' : 'Cancel Analysis'}
        </button>
      </div>
    </div>
  )
}
