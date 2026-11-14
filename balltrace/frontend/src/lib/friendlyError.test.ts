import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import {
  BACKEND_UNREACHABLE,
  GENERIC_ERROR,
  friendlyError,
  uploadTooLargeMessage,
} from './friendlyError'

// Anything that would tell a user about plumbing instead of what to do.
const TECHNICAL = /HTTP|\b[45]\d\d\b(?! MB)|stack trace|Traceback|undefined|\[object|TypeError|ECONN|fetch|JSON|zod|pydantic/i

describe('friendlyError', () => {
  let logged: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    logged = vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })
  afterEach(() => {
    logged.mockRestore()
  })

  it('shows the worded message the backend sent for a known failure', () => {
    const detail = 'BallTrace can\'t reach Ollama. Start Ollama, then try again.'
    const err = new ApiError(503, 'HTTP 503 calling /analyses/a1/start', 'ollama_unavailable', detail)
    expect(friendlyError(err, 'start analysis')).toBe(detail)
  })

  it('maps a network failure to "backend not running"', () => {
    const err = new ApiError(0, 'network error calling /analyses: TypeError: Failed to fetch')
    expect(friendlyError(err, 'load analyses')).toBe(BACKEND_UNREACHABLE)
  })

  it.each([
    [413, /too large/],
    [404, /couldn't find/],
    [422, /isn't valid|wasn't valid/],
    [500, /problem handling that/],
    [502, /problem handling that/],
    [418, /Something went wrong/],
  ])('maps a bare HTTP %i (no worded body) to plain language', (status, expected) => {
    const message = friendlyError(new ApiError(status, `HTTP ${status} calling /x`), 'load analysis')
    expect(message).toMatch(expected)
    expect(message).not.toMatch(TECHNICAL)
  })

  it('ignores a detail that arrived without a code, as it may be raw text', () => {
    const err = new ApiError(500, 'HTTP 500', null, 'Traceback (most recent call last): boom')
    const message = friendlyError(err, 'load analysis')
    expect(message).not.toContain('Traceback')
    expect(message).not.toMatch(TECHNICAL)
  })

  it('falls back to the generic message for anything else', () => {
    expect(friendlyError(new TypeError('x is undefined'), 'save settings')).toBe(GENERIC_ERROR)
    expect(friendlyError('a string', 'save settings')).toBe(GENERIC_ERROR)
    expect(friendlyError(null, 'save settings')).toBe(GENERIC_ERROR)
    expect(GENERIC_ERROR).toBe('Something went wrong. Try again.')
  })

  it('logs the raw error with where it happened, and never returns it', () => {
    const raw = new Error('connect ECONNREFUSED 127.0.0.1:8000')
    const message = friendlyError(raw, 'upload video')
    expect(message).not.toContain('ECONNREFUSED')
    expect(logged).toHaveBeenCalledTimes(1)
    expect(String(logged.mock.calls[0][0])).toContain('upload video')
    expect(logged.mock.calls[0][1]).toBe(raw)
  })

  it('words the upload limit with the number and a next step', () => {
    const message = uploadTooLargeMessage(500)
    expect(message).toContain('500 MB')
    expect(message).toMatch(/Trim or compress/)
    expect(message).not.toMatch(TECHNICAL)
  })
})
