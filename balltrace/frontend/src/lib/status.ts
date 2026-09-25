import type { AnalysisStatus } from '../types'

export const STATUS_LABEL: Record<AnalysisStatus, string> = {
  created: 'Created',
  uploaded: 'Ready',
  analyzing: 'Analyzing',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export const STATUS_COLOR: Record<AnalysisStatus, string> = {
  created: 'text-white/40',
  uploaded: 'text-sky-400',
  analyzing: 'text-amber-400',
  completed: 'text-neon-cyan',
  failed: 'text-red-400',
  cancelled: 'text-white/40',
}

export const STATUS_DOT: Record<AnalysisStatus, string> = {
  created: 'bg-white/30',
  uploaded: 'bg-sky-400',
  analyzing: 'bg-amber-400',
  completed: 'bg-neon-cyan',
  failed: 'bg-red-400',
  cancelled: 'bg-white/30',
}

export function formatDuration(seconds: number | null): string {
  if (seconds == null || Number.isNaN(seconds)) return '--:--'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}'`.replace("'", '')
}

export function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toFixed(2).padStart(5, '0')}`
}

export function formatRelativeDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  const diffMs = Date.now() - date.getTime()
  const diffMin = Math.floor(diffMs / 60000)
  if (diffMin < 1) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  const diffDay = Math.floor(diffHr / 24)
  if (diffDay < 7) return `${diffDay}d ago`
  return date.toLocaleDateString()
}
