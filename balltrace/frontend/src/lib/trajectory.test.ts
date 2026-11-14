import { describe, expect, it } from 'vitest'
import { DETECTIONS, detection } from '../test/fixtures'
import {
  currentDetectionAtTime,
  detectionsUpToTime,
  findAdjacentDetectionIndex,
  pixelPosition,
  segmentTrajectory,
} from './trajectory'

const point = (frame: number) => ({ detection: detection({ frame_index: frame }), px: frame, py: 0 })

describe('segmentTrajectory', () => {
  it('returns nothing for no points', () => {
    expect(segmentTrajectory([], 10)).toEqual([])
  })

  it('keeps nearby frames in one line and breaks where the gap is too big', () => {
    const segments = segmentTrajectory([0, 3, 6, 40, 43].map(point), 10)
    expect(segments.map((s) => s.map((p) => p.detection.frame_index))).toEqual([
      [0, 3, 6],
      [40, 43],
    ])
  })

  it('treats a gap equal to the limit as still connected', () => {
    expect(segmentTrajectory([0, 10].map(point), 10)).toHaveLength(1)
    expect(segmentTrajectory([0, 11].map(point), 10)).toHaveLength(2)
  })
})

describe('detectionsUpToTime', () => {
  it('keeps visible detections at or before the time, in time order', () => {
    const shuffled = [DETECTIONS[2], DETECTIONS[3], DETECTIONS[0], DETECTIONS[1]]
    expect(detectionsUpToTime(shuffled, 0.25).map((d) => d.id)).toEqual(['d0', 'd1', 'd2'])
    expect(detectionsUpToTime(shuffled, 0.1).map((d) => d.id)).toEqual(['d0', 'd1'])
  })
})

describe('currentDetectionAtTime', () => {
  it('returns the latest detection within the tolerance', () => {
    expect(currentDetectionAtTime(DETECTIONS, 0.15)?.id).toBe('d1')
  })
  it('returns null when the last detection is too old', () => {
    expect(currentDetectionAtTime(DETECTIONS, 0.9)).toBeNull()
  })
  it('returns null before the first detection', () => {
    expect(currentDetectionAtTime([detection({ timestamp: 1 })], 0.5)).toBeNull()
  })
})

describe('findAdjacentDetectionIndex', () => {
  it('finds the next and previous visible detection, skipping hidden ones', () => {
    expect(findAdjacentDetectionIndex(DETECTIONS, 0.1, 'next')).toBe(2)
    expect(findAdjacentDetectionIndex(DETECTIONS, 0.1, 'prev')).toBe(0)
  })
  it('returns -1 at either end or with nothing visible', () => {
    expect(findAdjacentDetectionIndex(DETECTIONS, 0.2, 'next')).toBe(-1)
    expect(findAdjacentDetectionIndex(DETECTIONS, 0, 'prev')).toBe(-1)
    expect(findAdjacentDetectionIndex([DETECTIONS[3]], 0, 'next')).toBe(-1)
  })
})

describe('pixelPosition', () => {
  it('prefers normalized coordinates scaled to the video size', () => {
    expect(pixelPosition(detection({ normalized_x: 0.5, normalized_y: 0.25 }), 200, 100)).toEqual({
      px: 100,
      py: 25,
    })
  })
  it('falls back to raw pixel coordinates', () => {
    const d = detection({ normalized_x: null, normalized_y: null, x: 12, y: 34 })
    expect(pixelPosition(d, 200, 100)).toEqual({ px: 12, py: 34 })
  })
  it('returns null when the ball has no position', () => {
    expect(pixelPosition(DETECTIONS[3], 200, 100)).toBeNull()
  })
})
