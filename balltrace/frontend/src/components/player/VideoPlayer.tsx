import { forwardRef, useEffect, useImperativeHandle } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import type { Analysis, Detection, TrajectorySettings } from '../../types'
import { useVideoPlayer } from '../../hooks/useVideoPlayer'
import { api } from '../../api/client'
import { TrajectoryCanvas } from './TrajectoryCanvas'
import { PlayerControls } from './PlayerControls'
import { findAdjacentDetectionIndex, currentDetectionAtTime } from '../../lib/trajectory'

export interface VideoPlayerHandle {
  seek: (time: number) => void
}

interface VideoPlayerProps {
  analysis: Analysis
  detections: Detection[]
  trajectorySettings: TrajectorySettings
  onTimeUpdate?: (currentTime: number, currentDetection: Detection | null) => void
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(function VideoPlayer(
  { analysis, detections, trajectorySettings, onTimeUpdate },
  ref,
) {
  const player = useVideoPlayer(analysis.fps)

  useImperativeHandle(ref, () => ({
    seek: (time: number) => {
      player.seek(time)
      if (!player.isPlaying) player.videoRef.current?.pause()
    },
  }))

  useEffect(() => {
    const current = currentDetectionAtTime(detections, player.currentTime)
    onTimeUpdate?.(player.currentTime, current)
  }, [player.currentTime, detections, onTimeUpdate])

  const handleJumpDetection = (direction: 'next' | 'prev') => {
    const idx = findAdjacentDetectionIndex(detections, player.currentTime, direction)
    if (idx >= 0) {
      player.pause()
      player.seek(detections[idx].timestamp)
    }
  }

  const videoUrl = analysis.video_url ? api.videoUrl(analysis.id) : null
  const hasDimensions = Boolean(analysis.video_width && analysis.video_height)

  return (
    <div className="flex h-full flex-col">
      <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-black/40 p-4">
        {!videoUrl && (
          <div className="flex flex-col items-center gap-2 text-white/40">
            <AlertTriangle size={22} />
            <p className="text-sm">No video available for this analysis.</p>
          </div>
        )}
        {videoUrl && (
          <div className="relative flex max-h-full max-w-full items-center justify-center">
            <video
              ref={player.attachVideo}
              src={videoUrl}
              className="max-h-full max-w-full rounded-md"
              playsInline
              preload="metadata"
            />
            {hasDimensions && (
              <TrajectoryCanvas
                videoRef={player.videoRef}
                detections={detections}
                videoWidth={analysis.video_width!}
                videoHeight={analysis.video_height!}
                settings={trajectorySettings}
              />
            )}
            {player.isBuffering && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                <Loader2 size={28} className="animate-spin text-white/70" />
              </div>
            )}
          </div>
        )}
      </div>

      {videoUrl && (
        <div className="border-t border-white/[0.08] bg-base-900/60">
          <PlayerControls
            isPlaying={player.isPlaying}
            currentTime={player.currentTime}
            duration={player.duration}
            speed={player.speed}
            onTogglePlay={player.togglePlay}
            onSeek={player.seek}
            onStepFrame={player.stepFrame}
            onJumpDetection={handleJumpDetection}
            onSetSpeed={player.setSpeed}
          />
        </div>
      )}
    </div>
  )
})
