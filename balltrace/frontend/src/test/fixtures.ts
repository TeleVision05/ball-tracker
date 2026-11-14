import type { Analysis, AnalysisSummary, Detection } from '../types'

export function detection(overrides: Partial<Detection> = {}): Detection {
  return {
    id: 'd0',
    frame_index: 0,
    timestamp: 0,
    ball_visible: true,
    x: 100,
    y: 50,
    normalized_x: 0.1,
    normalized_y: 0.05,
    confidence: 0.9,
    source_window: 0,
    ...overrides,
  }
}

/** Four detections: three visible (frames 0, 3, 6), one with the ball out of view (frame 9). */
export const DETECTIONS: Detection[] = [
  detection({ id: 'd0', frame_index: 0, timestamp: 0.0, confidence: 0.9 }),
  detection({ id: 'd1', frame_index: 3, timestamp: 0.1, confidence: 0.8 }),
  detection({ id: 'd2', frame_index: 6, timestamp: 0.2, confidence: 0.7 }),
  detection({
    id: 'd3',
    frame_index: 9,
    timestamp: 0.3,
    ball_visible: false,
    x: null,
    y: null,
    normalized_x: null,
    normalized_y: null,
    confidence: 0,
  }),
]

export function analysis(overrides: Partial<Analysis> = {}): Analysis {
  return {
    id: 'a1',
    name: 'Net session',
    video_path: '/data/a1/source.mp4',
    video_url: '/api/analyses/a1/video',
    video_width: 1920,
    video_height: 1080,
    fps: 30,
    duration: 4,
    frame_count: 120,
    created_at: '2026-01-01T10:00:00Z',
    status: 'completed',
    model: 'qwen3-vl:8b',
    error: null,
    detections: DETECTIONS,
    ...overrides,
  }
}

export function summary(overrides: Partial<AnalysisSummary> = {}): AnalysisSummary {
  const { detections, ...base } = analysis()
  return { ...base, detection_count: detections.length, ...overrides }
}
