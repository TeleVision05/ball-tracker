import { useEffect, useRef } from 'react'
import type { Detection, TrajectorySettings } from '../../types'
import { detectionsUpToTime, currentDetectionAtTime, pixelPosition, segmentTrajectory } from '../../lib/trajectory'
import type { TrajectoryPoint } from '../../lib/trajectory'

interface TrajectoryCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>
  detections: Detection[]
  videoWidth: number
  videoHeight: number
  settings: TrajectorySettings
}

const CYAN = '0, 255, 208'
const GREEN = '57, 255, 20'

export function TrajectoryCanvas({
  videoRef,
  detections,
  videoWidth,
  videoHeight,
  settings,
}: TrajectoryCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const detectionsRef = useRef(detections)
  detectionsRef.current = detections

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const resize = () => {
      const rect = video.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      canvas.style.width = `${rect.width}px`
      canvas.style.height = `${rect.height}px`
      canvas.width = Math.round(rect.width * dpr)
      canvas.height = Math.round(rect.height * dpr)
    }

    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(video)
    window.addEventListener('resize', resize)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', resize)
    }
  }, [videoRef])

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video || !videoWidth || !videoHeight) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const displayWidth = canvas.width / dpr
      const displayHeight = canvas.height / dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, displayWidth, displayHeight)

      const scaleX = displayWidth / videoWidth
      const scaleY = displayHeight / videoHeight
      const t = video.currentTime
      const s = settingsRef.current
      const all = detectionsRef.current

      if (s.showTrajectory || s.showMarkers) {
        const visible = detectionsUpToTime(all, t)
        const points: TrajectoryPoint[] = visible
          .map((d) => {
            const pos = pixelPosition(d, videoWidth, videoHeight)
            if (!pos) return null
            return { detection: d, px: pos.px * scaleX, py: pos.py * scaleY }
          })
          .filter((p): p is TrajectoryPoint => p !== null)

        if (s.showTrajectory) {
          const segments = segmentTrajectory(points, s.maxGapFrames)
          for (const segment of segments) {
            if (segment.length < 2) continue
            ctx.save()
            ctx.strokeStyle = `rgba(${CYAN}, 0.85)`
            ctx.lineWidth = s.lineThickness
            ctx.lineCap = 'round'
            ctx.lineJoin = 'round'
            ctx.shadowColor = `rgba(${CYAN}, 0.9)`
            ctx.shadowBlur = 10
            ctx.beginPath()
            ctx.moveTo(segment[0].px, segment[0].py)
            for (let i = 1; i < segment.length; i++) {
              ctx.lineTo(segment[i].px, segment[i].py)
            }
            ctx.stroke()
            ctx.restore()
          }
        }

        if (s.showMarkers) {
          for (const p of points) {
            ctx.save()
            const alpha = s.showConfidence ? Math.max(0.35, p.detection.confidence) : 0.9
            ctx.fillStyle = `rgba(${CYAN}, ${alpha})`
            ctx.shadowColor = `rgba(${CYAN}, 0.8)`
            ctx.shadowBlur = 6
            ctx.beginPath()
            ctx.arc(p.px, p.py, 3, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()
          }
        }
      }

      const current = currentDetectionAtTime(all, t)
      if (current) {
        const pos = pixelPosition(current, videoWidth, videoHeight)
        if (pos) {
          const cx = pos.px * scaleX
          const cy = pos.py * scaleY
          ctx.save()
          ctx.fillStyle = `rgba(${GREEN}, 0.95)`
          ctx.shadowColor = `rgba(${GREEN}, 1)`
          ctx.shadowBlur = 16
          ctx.beginPath()
          ctx.arc(cx, cy, 8, 0, Math.PI * 2)
          ctx.fill()
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
          ctx.lineWidth = 1.5
          ctx.stroke()
          ctx.restore()

          if (s.showConfidence) {
            ctx.save()
            ctx.font = '600 11px "IBM Plex Mono", monospace'
            ctx.fillStyle = 'rgba(255,255,255,0.9)'
            ctx.shadowColor = 'rgba(0,0,0,0.8)'
            ctx.shadowBlur = 4
            ctx.fillText(`${Math.round(current.confidence * 100)}%`, cx + 12, cy - 12)
            ctx.restore()
          }
        }
      }

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [videoRef, videoWidth, videoHeight])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
}
