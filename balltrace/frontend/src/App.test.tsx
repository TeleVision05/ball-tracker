import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { analysis, summary } from './test/fixtures'

vi.mock('./api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api/client')>()
  return {
    ...actual,
    api: {
      listAnalyses: vi.fn(),
      getAnalysis: vi.fn(),
      startAnalysis: vi.fn(),
      cancelAnalysis: vi.fn(),
      limits: vi.fn(),
      progressUrl: (id: string) => `/api/analyses/${id}/progress`,
      exportJsonUrl: (id: string) => `/api/analyses/${id}/export/json`,
      exportCsvUrl: (id: string) => `/api/analyses/${id}/export/csv`,
      getOllamaSettings: vi.fn(),
    },
  }
})

// The player needs a real <video> and <canvas>; its drawing is not under test here.
vi.mock('./components/player/VideoPlayer', () => ({
  VideoPlayer: () => <div data-testid="video-player" />,
}))

import { ApiError, api } from './api/client'

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.mocked(api.limits).mockResolvedValue({ max_upload_mb: 500 })
  vi.mocked(api.listAnalyses).mockResolvedValue([summary()])
  vi.mocked(api.getAnalysis).mockResolvedValue(analysis())
})

describe('App: results', () => {
  it('lists saved analyses and renders the stats and timeline of a finished one', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: /Net session/ }))

    // Right sidebar, from the fixture: 4 detections, 3 visible (75%), mean confidence 80%.
    expect(await screen.findByTestId('video-player')).toBeInTheDocument()
    const stats = screen.getByText('Detection Stats').closest('section') as HTMLElement
    expect(within(stats).getByText('4')).toBeInTheDocument()
    expect(within(stats).getByText('75%')).toBeInTheDocument()
    expect(within(stats).getByText('80%')).toBeInTheDocument()

    // Timeline lists only the three frames where the ball is visible.
    const timeline = screen.getByText('Detection Timeline').closest('section') as HTMLElement
    expect(within(timeline).getAllByText(/^frame \d+$/).map((el) => el.textContent)).toEqual([
      'frame 0',
      'frame 3',
      'frame 6',
    ])
    // Once in the list, once in the sidebar header.
    expect(screen.getAllByText('Completed')).toHaveLength(2)
    expect(screen.getByText(/qwen3-vl:8b/)).toBeInTheDocument()
  })

  it('says so when the analysis finished but found no ball', async () => {
    vi.mocked(api.getAnalysis).mockResolvedValue(
      analysis({ detections: [analysis().detections[3]] }),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: /Net session/ }))
    expect(await screen.findByText(/no visible-ball detections/)).toBeInTheDocument()
  })

  it('offers upload and start for an analysis without results yet', async () => {
    vi.mocked(api.getAnalysis).mockResolvedValue(
      analysis({ status: 'uploaded', detections: [] }),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: /Net session/ }))
    expect(await screen.findByRole('button', { name: 'Start Analysis' })).toBeInTheDocument()
    expect(screen.getByText('No detections recorded.')).toBeInTheDocument()
  })
})

describe('App: errors go through the friendly helper', () => {
  it('shows plain language when the list cannot be loaded', async () => {
    vi.mocked(api.listAnalyses).mockRejectedValue(
      new ApiError(0, 'network error calling /analyses: TypeError: Failed to fetch'),
    )
    render(<App />)
    expect(await screen.findByText(/Can't reach the BallTrace backend/)).toBeInTheDocument()
    expect(screen.queryByText(/Failed to fetch|TypeError|network error/)).not.toBeInTheDocument()
  })

  it('shows the backend worded message and a Retry when one analysis cannot be opened', async () => {
    const detail =
      'We couldn\'t find that analysis. It may have been deleted. Refresh the list and try again.'
    vi.mocked(api.getAnalysis).mockRejectedValue(
      new ApiError(404, 'HTTP 404 calling /analyses/a1', 'analysis_not_found', detail),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: /Net session/ }))
    expect(await screen.findByText(detail)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('falls back to "Something went wrong" for an unexpected failure, and logs the raw one', async () => {
    const raw = new TypeError("Cannot read properties of undefined (reading 'map')")
    vi.mocked(api.getAnalysis).mockRejectedValue(raw)
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: /Net session/ }))
    expect(await screen.findByText('Something went wrong. Try again.')).toBeInTheDocument()
    expect(screen.queryByText(/Cannot read properties/)).not.toBeInTheDocument()
    expect(console.error).toHaveBeenCalledWith(expect.stringContaining('load analysis'), raw)
  })
})
