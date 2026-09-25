import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  SkipBack,
  SkipForward,
} from 'lucide-react'
import type { PlaybackSpeed } from '../../types'
import { SPEED_OPTIONS } from '../../hooks/useVideoPlayer'
import { formatTimestamp } from '../../lib/status'

interface PlayerControlsProps {
  isPlaying: boolean
  currentTime: number
  duration: number
  speed: PlaybackSpeed
  onTogglePlay: () => void
  onSeek: (time: number) => void
  onStepFrame: (direction: 1 | -1) => void
  onJumpDetection: (direction: 'next' | 'prev') => void
  onSetSpeed: (speed: PlaybackSpeed) => void
}

export function PlayerControls({
  isPlaying,
  currentTime,
  duration,
  speed,
  onTogglePlay,
  onSeek,
  onStepFrame,
  onJumpDetection,
  onSetSpeed,
}: PlayerControlsProps) {
  return (
    <div className="flex flex-col gap-2.5 px-4 py-3">
      <input
        type="range"
        min={0}
        max={duration || 0}
        step={0.001}
        value={Math.min(currentTime, duration || 0)}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="trajectory-seek h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-neon-cyan"
        style={{
          background: `linear-gradient(to right, #00ffd0 ${
            duration ? (currentTime / duration) * 100 : 0
          }%, rgba(255,255,255,0.1) ${duration ? (currentTime / duration) * 100 : 0}%)`,
        }}
      />

      <div className="flex items-center gap-1">
        <span className="w-[86px] font-mono text-xs text-white/45">
          {formatTimestamp(currentTime)} / {formatTimestamp(duration)}
        </span>

        <div className="flex flex-1 items-center justify-center gap-1">
          <button
            onClick={() => onJumpDetection('prev')}
            className="btn-ghost"
            title="Jump to previous detection"
          >
            <SkipBack size={15} />
          </button>
          <button onClick={() => onStepFrame(-1)} className="btn-ghost" title="Previous frame">
            <ChevronLeft size={16} />
          </button>
          <button
            onClick={onTogglePlay}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-neon-cyan text-base-950 shadow-glow transition-transform hover:scale-105"
          >
            {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
          </button>
          <button onClick={() => onStepFrame(1)} className="btn-ghost" title="Next frame">
            <ChevronRight size={16} />
          </button>
          <button
            onClick={() => onJumpDetection('next')}
            className="btn-ghost"
            title="Jump to next detection"
          >
            <SkipForward size={15} />
          </button>
        </div>

        <div className="flex w-[86px] justify-end gap-0.5">
          {SPEED_OPTIONS.map((opt) => (
            <button
              key={opt}
              onClick={() => onSetSpeed(opt)}
              className={`rounded px-1.5 py-1 text-[11px] font-mono font-medium transition-colors ${
                speed === opt ? 'bg-white/10 text-neon-cyan' : 'text-white/35 hover:text-white/70'
              }`}
            >
              {opt}x
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
