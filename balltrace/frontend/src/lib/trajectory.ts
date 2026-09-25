import type { Detection } from '../types'

export interface TrajectoryPoint {
  detection: Detection
  px: number
  py: number
}

/**
 * Splits a time-ordered list of visible-ball points into contiguous segments,
 * breaking the line whenever the gap between consecutive frames exceeds
 * `maxGapFrames`. Used so the overlay never draws a line across a stretch of
 * frames where the ball was not detected.
 */
export function segmentTrajectory(
  points: TrajectoryPoint[],
  maxGapFrames: number,
): TrajectoryPoint[][] {
  if (points.length === 0) return []

  const segments: TrajectoryPoint[][] = []
  let current: TrajectoryPoint[] = [points[0]]

  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]
    const curr = points[i]
    const frameDiff = Math.abs(curr.detection.frame_index - prev.detection.frame_index)
    if (frameDiff > maxGapFrames) {
      segments.push(current)
      current = [curr]
    } else {
      current.push(curr)
    }
  }
  segments.push(current)
  return segments
}

/** Detections visible at/before time T, sorted by timestamp ascending. */
export function detectionsUpToTime(detections: Detection[], t: number): Detection[] {
  return detections
    .filter((d) => d.ball_visible && d.x != null && d.y != null && d.timestamp <= t)
    .sort((a, b) => a.timestamp - b.timestamp)
}

/** The detection whose timestamp is closest to (and at/before) time T. */
export function currentDetectionAtTime(
  detections: Detection[],
  t: number,
  toleranceSeconds = 0.12,
): Detection | null {
  const visible = detectionsUpToTime(detections, t)
  if (visible.length === 0) return null
  const last = visible[visible.length - 1]
  if (t - last.timestamp > toleranceSeconds) return null
  return last
}

/** Finds the nearest detection index to a given time, for next/prev jump controls. */
export function findAdjacentDetectionIndex(
  detections: Detection[],
  currentTime: number,
  direction: 'next' | 'prev',
): number {
  const visible = detections
    .map((d, idx) => ({ d, idx }))
    .filter((entry) => entry.d.ball_visible)
    .sort((a, b) => a.d.timestamp - b.d.timestamp)

  if (visible.length === 0) return -1

  if (direction === 'next') {
    const found = visible.find((entry) => entry.d.timestamp > currentTime + 1e-3)
    return found ? found.idx : -1
  }

  const reversed = [...visible].reverse()
  const found = reversed.find((entry) => entry.d.timestamp < currentTime - 1e-3)
  return found ? found.idx : -1
}

export function pixelPosition(
  detection: Detection,
  videoWidth: number,
  videoHeight: number,
): { px: number; py: number } | null {
  if (detection.normalized_x != null && detection.normalized_y != null) {
    return { px: detection.normalized_x * videoWidth, py: detection.normalized_y * videoHeight }
  }
  if (detection.x != null && detection.y != null) {
    return { px: detection.x, py: detection.y }
  }
  return null
}
