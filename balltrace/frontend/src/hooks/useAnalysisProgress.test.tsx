import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ProgressEvent } from '../types'
import { useAnalysisProgress } from './useAnalysisProgress'

class FakeEventSource {
  static instances: FakeEventSource[] = []
  url: string
  closed = false
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }
  close() {
    this.closed = true
  }
  emit(data: unknown) {
    this.onmessage?.({ data: typeof data === 'string' ? data : JSON.stringify(data) })
  }
}

function event(overrides: Partial<ProgressEvent> = {}): ProgressEvent {
  return {
    analysis_id: 'a1',
    status: 'analyzing',
    current_window: 2,
    total_windows: 8,
    detections_count: 1,
    elapsed_seconds: 3,
    eta_seconds: 9,
    message: 'frame 2/8',
    log: [],
    ...overrides,
  }
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useAnalysisProgress', () => {
  it('does not connect until the analysis is active', () => {
    renderHook(() => useAnalysisProgress('a1', false))
    expect(FakeEventSource.instances).toHaveLength(0)
  })

  it('connects to the progress stream and exposes each update', () => {
    const { result } = renderHook(() => useAnalysisProgress('a1', true))
    const source = FakeEventSource.instances[0]
    expect(source.url).toBe('/api/analyses/a1/progress')

    act(() => source.onopen?.())
    expect(result.current.connected).toBe(true)

    act(() => source.emit(event()))
    expect(result.current.progress?.current_window).toBe(2)
    expect(result.current.streamError).toBeNull()
  })

  it('closes the stream and calls back once a terminal status arrives', () => {
    const onTerminal = vi.fn()
    const { result } = renderHook(() => useAnalysisProgress('a1', true, onTerminal))
    const source = FakeEventSource.instances[0]

    act(() => source.emit(event({ status: 'failed', message: 'x' })))
    expect(onTerminal).toHaveBeenCalledTimes(1)
    expect(source.closed).toBe(true)
    expect(result.current.connected).toBe(false)
  })

  it('closes the stream when the component unmounts', () => {
    const { unmount } = renderHook(() => useAnalysisProgress('a1', true))
    unmount()
    expect(FakeEventSource.instances[0].closed).toBe(true)
  })

  it('explains a dropped connection and an unreadable update without technical words', () => {
    const { result } = renderHook(() => useAnalysisProgress('a1', true))
    const source = FakeEventSource.instances[0]

    act(() => source.onerror?.())
    expect(result.current.streamError).toMatch(/analysis keeps running/)
    expect(result.current.streamError).not.toMatch(/stream|SSE|EventSource/i)

    act(() => source.emit('{not json'))
    expect(result.current.streamError).toMatch(/couldn't read/)
    expect(result.current.streamError).not.toMatch(/JSON|SyntaxError|malformed/i)
  })
})
