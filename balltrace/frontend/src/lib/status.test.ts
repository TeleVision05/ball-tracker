import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { formatDuration, formatRelativeDate, formatTimestamp } from './status'

describe('formatTimestamp', () => {
  it('formats seconds as m:ss.ss', () => {
    expect(formatTimestamp(0)).toBe('0:00.00')
    expect(formatTimestamp(7.5)).toBe('0:07.50')
    expect(formatTimestamp(83.256)).toBe('1:23.26')
  })
})

describe('formatDuration', () => {
  it('formats whole seconds as m:ss', () => {
    expect(formatDuration(65)).toBe('1:05')
    expect(formatDuration(9)).toBe('0:09')
  })
  it('shows a placeholder for a missing duration', () => {
    expect(formatDuration(null)).toBe('--:--')
    expect(formatDuration(Number.NaN)).toBe('--:--')
  })
})

describe('formatRelativeDate', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-06-15T12:00:00Z'))
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it.each([
    ['2026-06-15T11:59:40Z', 'just now'],
    ['2026-06-15T11:30:00Z', '30m ago'],
    ['2026-06-15T07:00:00Z', '5h ago'],
    ['2026-06-12T12:00:00Z', '3d ago'],
  ])('%s -> %s', (iso, expected) => {
    expect(formatRelativeDate(iso)).toBe(expected)
  })

  it('returns an empty string for an unreadable date', () => {
    expect(formatRelativeDate('not a date')).toBe('')
  })
})
