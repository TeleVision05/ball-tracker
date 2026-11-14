import type {
  Analysis,
  AnalysisSummary,
  Detection,
  ExportVideoRequest,
  ExportVideoResponse,
  HealthResponse,
  Limits,
  OllamaSettings,
  OllamaSettingsUpdate,
  StartAnalysisRequest,
} from '../types'

const API_BASE = '/api'

/**
 * A failed API call. `message` is for logs only. `detail` and `code` are set
 * only when the backend sent a worded error; show them through `friendlyError`.
 */
export class ApiError extends Error {
  status: number
  code: string | null
  detail: string | null
  constructor(status: number, message: string, code: string | null = null, detail: string | null = null) {
    super(message)
    this.status = status
    this.code = code
    this.detail = detail
    this.name = 'ApiError'
  }
}

async function readErrorBody(res: Response): Promise<{ code: string | null; detail: string | null }> {
  try {
    const body = await res.json()
    if (typeof body?.code === 'string' && typeof body?.detail === 'string') {
      return { code: body.code, detail: body.detail }
    }
  } catch {
    // no JSON body (for example a proxy error page)
  }
  return { code: null, detail: null }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, init)
  } catch (cause) {
    throw new ApiError(0, `network error calling ${path}: ${String(cause)}`)
  }
  if (!res.ok) {
    const { code, detail } = await readErrorBody(res)
    throw new ApiError(res.status, `HTTP ${res.status} calling ${path}`, code, detail)
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

  limits(): Promise<Limits> {
    return request('/limits')
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
