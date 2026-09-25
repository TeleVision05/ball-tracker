import { RotateCcw, XCircle, Eye, EyeOff, Target, Gauge } from 'lucide-react'
import type { Analysis, Detection, TrajectorySettings } from '../../types'
import { DetectionList } from '../analysis/DetectionList'
import { ExportPanel } from '../analysis/ExportPanel'
import { STATUS_COLOR, STATUS_LABEL, formatTimestamp } from '../../lib/status'

interface RightSidebarProps {
  analysis: Analysis
  detections: Detection[]
  currentTime: number
  currentDetection: Detection | null
  trajectorySettings: TrajectorySettings
  onTrajectorySettingsChange: (patch: Partial<TrajectorySettings>) => void
  onSeek: (time: number) => void
  onClear: () => void
  onRerun: () => void
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/[0.03] px-3 py-2.5">
      <p className="label-text">{label}</p>
      <p className="mt-1 font-mono text-base font-semibold text-white">{value}</p>
    </div>
  )
}

function ToggleRow({
  icon,
  label,
  checked,
  onChange,
}: {
  icon: React.ReactNode
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-sm text-white/70 transition-colors hover:bg-white/[0.05] hover:text-white"
    >
      <span className="flex items-center gap-2">
        {icon}
        {label}
      </span>
      <span
        className={`relative h-4 w-7 rounded-full transition-colors ${
          checked ? 'bg-neon-cyan/80' : 'bg-white/15'
        }`}
      >
        <span
          className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
            checked ? 'translate-x-3.5' : 'translate-x-0.5'
          }`}
        />
      </span>
    </button>
  )
}

export function RightSidebar({
  analysis,
  detections,
  currentTime,
  currentDetection,
  trajectorySettings,
  onTrajectorySettingsChange,
  onSeek,
  onClear,
  onRerun,
}: RightSidebarProps) {
  const visibleDetections = detections.filter((d) => d.ball_visible)
  const avgConfidence =
    visibleDetections.length > 0
      ? visibleDetections.reduce((sum, d) => sum + d.confidence, 0) / visibleDetections.length
      : 0
  const visibilityPct =
    detections.length > 0 ? Math.round((visibleDetections.length / detections.length) * 100) : 0

  const canRerun = analysis.status !== 'analyzing' && Boolean(analysis.video_url)

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-4">
      <section>
        <p className="label-text mb-2">Analysis</p>
        <h2 className="truncate font-display text-base font-semibold text-white">
          {analysis.name}
        </h2>
        <div className="mt-1.5 flex items-center gap-1.5 text-xs">
          <span className={`font-medium ${STATUS_COLOR[analysis.status]}`}>
            {STATUS_LABEL[analysis.status]}
          </span>
          {analysis.model && <span className="text-white/30">· {analysis.model}</span>}
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-white/45">
          <div>
            <span className="text-white/25">Resolution</span>
            <p className="font-mono text-white/70">
              {analysis.video_width && analysis.video_height
                ? `${analysis.video_width}×${analysis.video_height}`
                : '—'}
            </p>
          </div>
          <div>
            <span className="text-white/25">FPS</span>
            <p className="font-mono text-white/70">{analysis.fps?.toFixed(2) ?? '—'}</p>
          </div>
          <div>
            <span className="text-white/25">Duration</span>
            <p className="font-mono text-white/70">
              {analysis.duration ? `${analysis.duration.toFixed(1)}s` : '—'}
            </p>
          </div>
          <div>
            <span className="text-white/25">Frames</span>
            <p className="font-mono text-white/70">{analysis.frame_count ?? '—'}</p>
          </div>
        </div>
        {analysis.error && (
          <p className="mt-3 rounded-md bg-red-500/10 px-2.5 py-2 text-xs text-red-300">
            {analysis.error}
          </p>
        )}
      </section>

      <section className="border-t border-white/[0.08] pt-4">
        <p className="label-text mb-2">Current Ball</p>
        {currentDetection ? (
          <div className="glass-panel rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm text-neon-cyan">
                {formatTimestamp(currentDetection.timestamp)}
              </span>
              <span className="font-mono text-xs text-white/50">
                {Math.round(currentDetection.confidence * 100)}% conf
              </span>
            </div>
            <div className="mt-2 flex items-center gap-1.5 font-mono text-xs text-white/60">
              <Target size={12} className="text-neon-green" />
              x: {currentDetection.x?.toFixed(1) ?? '—'} y: {currentDetection.y?.toFixed(1) ?? '—'}
            </div>
            <div className="mt-1 flex items-center gap-1.5 font-mono text-xs text-white/40">
              <Gauge size={12} />
              frame {currentDetection.frame_index} · window {currentDetection.source_window}
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-white/10 px-3 py-3 text-xs text-white/35">
            No ball detected at {formatTimestamp(currentTime)}
          </div>
        )}
      </section>

      <section className="border-t border-white/[0.08] pt-4">
        <p className="label-text mb-2">Detection Stats</p>
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Total" value={String(detections.length)} />
          <Stat label="Visible" value={`${visibilityPct}%`} />
          <Stat label="Avg Conf" value={`${Math.round(avgConfidence * 100)}%`} />
        </div>
      </section>

      <section className="border-t border-white/[0.08] pt-4">
        <p className="label-text mb-2">Trajectory Overlay</p>
        <div className="flex flex-col gap-0.5">
          <ToggleRow
            icon={<Eye size={14} />}
            label="Show trajectory"
            checked={trajectorySettings.showTrajectory}
            onChange={(v) => onTrajectorySettingsChange({ showTrajectory: v })}
          />
          <ToggleRow
            icon={<Target size={14} />}
            label="Show markers"
            checked={trajectorySettings.showMarkers}
            onChange={(v) => onTrajectorySettingsChange({ showMarkers: v })}
          />
          <ToggleRow
            icon={<Gauge size={14} />}
            label="Show confidence"
            checked={trajectorySettings.showConfidence}
            onChange={(v) => onTrajectorySettingsChange({ showConfidence: v })}
          />
        </div>

        <div className="mt-3 flex flex-col gap-3">
          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs text-white/50">Line thickness</span>
              <span className="font-mono text-xs text-white/70">
                {trajectorySettings.lineThickness}px
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={8}
              step={1}
              value={trajectorySettings.lineThickness}
              onChange={(e) =>
                onTrajectorySettingsChange({ lineThickness: Number(e.target.value) })
              }
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-neon-cyan"
            />
          </div>
          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs text-white/50">Max gap (frames)</span>
              <span className="font-mono text-xs text-white/70">
                {trajectorySettings.maxGapFrames}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={60}
              step={1}
              value={trajectorySettings.maxGapFrames}
              onChange={(e) =>
                onTrajectorySettingsChange({ maxGapFrames: Number(e.target.value) })
              }
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-neon-cyan"
            />
          </div>
        </div>

        <div className="mt-3 flex gap-2">
          <button
            onClick={onRerun}
            disabled={!canRerun}
            className="btn-secondary flex-1"
            title="Re-run analysis on this video"
          >
            <RotateCcw size={14} />
            Re-run
          </button>
          <button onClick={onClear} className="btn-secondary flex-1">
            <XCircle size={14} />
            Clear
          </button>
        </div>
      </section>

      <section className="border-t border-white/[0.08] pt-4">
        <p className="label-text mb-2">Detection Timeline</p>
        {detections.length === 0 ? (
          <div className="flex items-center gap-2 px-1 py-3 text-sm text-white/30">
            <EyeOff size={14} />
            No detections recorded.
          </div>
        ) : (
          <DetectionList
            detections={detections}
            currentDetectionId={currentDetection?.id ?? null}
            onSeek={onSeek}
          />
        )}
      </section>

      <section className="border-t border-white/[0.08] pt-4">
        <p className="label-text mb-2">Export</p>
        <ExportPanel analysisId={analysis.id} trajectorySettings={trajectorySettings} />
      </section>
    </div>
  )
}
