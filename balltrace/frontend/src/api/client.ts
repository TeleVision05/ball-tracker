import type {
  Analysis,
  AnalysisSummary,
  Detection,
  ExportVideoRequest,
  ExportVideoResponse,
  HealthResponse,
  OllamaSettings,
  OllamaSettingsUpdate,
  StartAnalysisRequest,
} from '../types'

const API_BASE = '/api'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'Could not reach the BallTrace backend. Is it running on port 8000?')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body?.detail ?? detail
    } catch {
      // response had no JSON body; fall back to statusText
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

function jsonHeaders(): HeadersInit {
  return { 'Content-Type': 'application/json' }
}

export const api = {
  health(): Promise<HealthResponse> {
    return request('/health')
  },

  getOllamaSettings(): Promise<OllamaSettings> {
    return request('/settings/ollama')
  },

  updateOllamaSettings(payload: OllamaSettingsUpdate): Promise<OllamaSettings> {
    return request('/settings/ollama', {
      method: 'PUT',
      headers: jsonHeaders(),
      body: JSON.stringify(payload),
    })
  },

  listAnalyses(): Promise<AnalysisSummary[]> {
    return request('/analyses')
  },

  getAnalysis(id: string): Promise<Analysis> {
    return request(`/analyses/${id}`)
  },

  createAnalysis(name: string): Promise<Analysis> {
    return request('/analyses', {
      method: 'POST',
      headers: jsonHeaders(),
      body: JSON.stringify({ name }),
    })
  },

  uploadVideo(id: string, file: File): Promise<Analysis> {
    const form = new FormData()
    form.append('file', file)
    return request(`/analyses/${id}/upload`, {
      method: 'POST',
      body: form,
    })
  },

  startAnalysis(id: string, payload: StartAnalysisRequest = {}): Promise<{ status: string }> {
    return request(`/analyses/${id}/start`, {
      method: 'POST',
      headers: jsonHeaders(),
      body: JSON.stringify(payload),
    })
  },

  cancelAnalysis(id: string): Promise<{ status: string }> {
    return request(`/analyses/${id}/cancel`, { method: 'POST' })
  },

  getDetections(id: string): Promise<Detection[]> {
    return request(`/analyses/${id}/detections`)
  },

  videoUrl(id: string): string {
    return `${API_BASE}/analyses/${id}/video`
  },

  progressUrl(id: string): string {
    return `${API_BASE}/analyses/${id}/progress`
  },

  exportJsonUrl(id: string): string {
    return `${API_BASE}/analyses/${id}/export/json`
  },

  exportCsvUrl(id: string): string {
    return `${API_BASE}/analyses/${id}/export/csv`
  },

  exportVideo(id: string, payload: ExportVideoRequest = {}): Promise<ExportVideoResponse> {
    return request(`/analyses/${id}/export/video`, {
      method: 'POST',
      headers: jsonHeaders(),
      body: JSON.stringify(payload),
    })
  },

  exportVideoFileUrl(id: string): string {
    return `${API_BASE}/analyses/${id}/export/video/file`
  },
}
