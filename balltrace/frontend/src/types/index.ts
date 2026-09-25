// Types mirror balltrace/API_CONTRACT.md exactly (snake_case fields, matching
// the FastAPI pydantic models in backend/app/models/schemas.py).

export type AnalysisStatus =
  | 'created'
  | 'uploaded'
  | 'analyzing'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface Detection {
  id: string
  frame_index: number
  timestamp: number
  ball_visible: boolean
  x: number | null
  y: number | null
  normalized_x: number | null
  normalized_y: number | null
  confidence: number
  source_window: number
}

export interface AnalysisBase {
  id: string
  name: string
  video_path: string | null
  video_url: string | null
  video_width: number | null
  video_height: number | null
  fps: number | null
  duration: number | null
  frame_count: number | null
  created_at: string
  status: AnalysisStatus
  model: string | null
  error: string | null
}

export interface Analysis extends AnalysisBase {
  detections: Detection[]
}

export interface AnalysisSummary extends AnalysisBase {
  detection_count: number
}

export interface CreateAnalysisRequest {
  name: string
}

export interface StartAnalysisRequest {
  model?: string
  window_size?: number
  window_overlap?: number
}

export interface ProgressEvent {
  analysis_id: string
  status: AnalysisStatus | 'unknown'
  current_window: number
  total_windows: number
  detections_count: number
  elapsed_seconds: number
  eta_seconds: number | null
  message: string
  log: string[]
}

export interface HealthResponse {
  status: string
  ollama_available: boolean
  ollama_url: string
  ollama_model: string
  ffmpeg_available: boolean
}

export interface OllamaSettings {
  base_url: string
  model: string
  window_size: number
  window_overlap: number
  frame_stride: number
  available: boolean
  error: string | null
}

export interface OllamaSettingsUpdate {
  base_url?: string
  model?: string
  window_size?: number
  window_overlap?: number
  frame_stride?: number
}

export interface ExportVideoRequest {
  line_thickness?: number
  max_gap_frames?: number
  show_markers?: boolean
}

export interface ExportVideoResponse {
  job_id: string
  status: string
  download_url?: string | null
}

export type PlaybackSpeed = 0.25 | 0.5 | 1 | 1.5 | 2

export interface TrajectorySettings {
  showTrajectory: boolean
  showMarkers: boolean
  showConfidence: boolean
  lineThickness: number
  maxGapFrames: number
}
