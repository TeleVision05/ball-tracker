import { useEffect, useRef } from 'react'
import { Circle, Target } from 'lucide-react'
import type { Detection } from '../../types'
import { formatTimestamp } from '../../lib/status'

interface DetectionListProps {
  detections: Detection[]
  currentDetectionId: string | null
  onSeek: (timestamp: number) => void
}

export function DetectionList({ detections, currentDetectionId, onSeek }: DetectionListProps) {
  const activeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [currentDetectionId])

  const visible = detections.filter((d) => d.ball_visible)

  if (detections.length === 0) {
    return <div className="px-1 py-4 text-sm text-white/30">No detections yet.</div>
  }

  if (visible.length === 0) {
    return (
      <div className="px-1 py-4 text-sm text-white/30">
        Analysis finished with no visible-ball detections.
      </div>
    )
  }

  return (
    <div className="flex max-h-64 flex-col gap-0.5 overflow-y-auto pr-1">
      {visible.map((d) => {
        const isActive = d.id === currentDetectionId
        return (
          <button
            key={d.id}
            ref={isActive ? activeRef : undefined}
            onClick={() => onSeek(d.timestamp)}
            className={`flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors ${
              isActive
                ? 'bg-neon-cyan/10 text-neon-cyan ring-1 ring-inset ring-neon-cyan/30'
                : 'text-white/60 hover:bg-white/[0.05] hover:text-white/85'
            }`}
          >
            <span className="flex items-center gap-1.5 font-mono">
              {isActive ? <Target size={12} /> : <Circle size={7} fill="currentColor" />}
              {formatTimestamp(d.timestamp)}
            </span>
            <span className="text-white/35">frame {d.frame_index}</span>
            <span className="font-mono tabular-nums">{Math.round(d.confidence * 100)}%</span>
          </button>
        )
      })}
    </div>
  )
}
