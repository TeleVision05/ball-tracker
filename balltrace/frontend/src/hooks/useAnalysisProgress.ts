import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { ProgressEvent } from '../types'

interface UseAnalysisProgressResult {
  progress: ProgressEvent | null
  connected: boolean
  streamError: string | null
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled'])

/**
 * Subscribes to GET /api/analyses/{id}/progress via SSE while `active` is
 * true. Automatically closes the stream once a terminal status arrives.
 */
export function useAnalysisProgress(
  analysisId: string | null,
  active: boolean,
  onTerminal?: (event: ProgressEvent) => void,
): UseAnalysisProgressResult {
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [connected, setConnected] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)
  const onTerminalRef = useRef(onTerminal)
  onTerminalRef.current = onTerminal

  useEffect(() => {
    if (!analysisId || !active) {
      setConnected(false)
      return
    }

    setStreamError(null)
    const source = new EventSource(api.progressUrl(analysisId))

    source.onopen = () => setConnected(true)

    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ProgressEvent
        setProgress(data)
        if (TERMINAL_STATUSES.has(data.status)) {
          onTerminalRef.current?.(data)
          source.close()
          setConnected(false)
        }
      } catch {
        setStreamError('Received malformed progress event from server.')
      }
    }

    source.onerror = () => {
      setConnected(false)
      setStreamError('Lost connection to progress stream.')
    }

    return () => {
      source.close()
      setConnected(false)
    }
  }, [analysisId, active])

  return { progress, connected, streamError }
}
