import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { analysis } from '../../test/fixtures'
import { AnalysisSetupPanel } from './AnalysisSetupPanel'

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>()
  return { ...actual, api: { uploadVideo: vi.fn(), startAnalysis: vi.fn(), limits: vi.fn() } }
})

import { ApiError, api } from '../../api/client'

const noVideo = analysis({ status: 'created', video_url: null, detections: [] })
const ready = analysis({ status: 'uploaded', detections: [] })

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.mocked(api.limits).mockResolvedValue({ max_upload_mb: 500 })
})

describe('AnalysisSetupPanel', () => {
  it('asks for a video when the analysis has none, then reports the upload', async () => {
    vi.mocked(api.uploadVideo).mockResolvedValue(ready)
    const onUploaded = vi.fn()
    const user = userEvent.setup()
    const { container } = render(
      <AnalysisSetupPanel analysis={noVideo} onUploaded={onUploaded} onStarted={vi.fn()} />,
    )
    expect(screen.getByText('Upload a video')).toBeInTheDocument()

    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    await user.upload(input, new File(['x'], 'bowl.mp4', { type: 'video/mp4' }))
    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith(ready))
    expect(api.uploadVideo).toHaveBeenCalledWith('a1', expect.any(File))
  })

  it('explains an oversized upload in words, with the limit', async () => {
    vi.mocked(api.uploadVideo).mockRejectedValue(
      new ApiError(
        413,
        'HTTP 413 calling /upload',
        'upload_too_large',
        'That video is larger than the 500 MB limit. Trim or compress it, then upload it again.',
      ),
    )
    const user = userEvent.setup()
    const { container } = render(
      <AnalysisSetupPanel analysis={noVideo} onUploaded={vi.fn()} onStarted={vi.fn()} />,
    )
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    await user.upload(input, new File(['x'], 'huge.mp4', { type: 'video/mp4' }))
    expect(await screen.findByText(/larger than the 500 MB limit/)).toBeInTheDocument()
    expect(screen.queryByText(/413/)).not.toBeInTheDocument()
  })

  it('shows what is ready and starts the analysis', async () => {
    vi.mocked(api.startAnalysis).mockResolvedValue({ status: 'analyzing' })
    const onStarted = vi.fn()
    const user = userEvent.setup()
    render(<AnalysisSetupPanel analysis={ready} onUploaded={vi.fn()} onStarted={onStarted} />)

    expect(screen.getByText('Ready to analyze')).toBeInTheDocument()
    expect(screen.getByText(/120 frames · 30\.0 fps · 4\.0s/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Start Analysis' }))
    await waitFor(() => expect(onStarted).toHaveBeenCalled())
  })

  it('tells the user how to fix Ollama when starting fails, without technical detail', async () => {
    const detail = 'The model "qwen3-vl:8b" isn\'t installed in Ollama. Install it with `ollama pull qwen3-vl:8b` or choose another model in Settings.'
    vi.mocked(api.startAnalysis).mockRejectedValue(
      new ApiError(503, 'HTTP 503 calling /start', 'ollama_model_missing', detail),
    )
    const user = userEvent.setup()
    render(<AnalysisSetupPanel analysis={ready} onUploaded={vi.fn()} onStarted={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: 'Start Analysis' }))
    expect(await screen.findByText(detail)).toBeInTheDocument()
    expect(screen.queryByText(/503/)).not.toBeInTheDocument()
  })

  it('shows the stored failure message and offers a retry after a failed run', () => {
    const failed = analysis({
      status: 'failed',
      detections: [],
      error: "BallTrace can't reach Ollama. Start Ollama, then try again.",
    })
    render(<AnalysisSetupPanel analysis={failed} onUploaded={vi.fn()} onStarted={vi.fn()} />)
    expect(screen.getByText('Analysis did not complete')).toBeInTheDocument()
    expect(screen.getByText(/can't reach Ollama/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry Analysis' })).toBeInTheDocument()
  })
})
