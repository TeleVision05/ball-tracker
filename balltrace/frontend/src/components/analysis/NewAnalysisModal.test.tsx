import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { analysis } from '../../test/fixtures'
import { NewAnalysisModal } from './NewAnalysisModal'

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>()
  return {
    ...actual,
    api: { createAnalysis: vi.fn(), uploadVideo: vi.fn(), limits: vi.fn() },
  }
})

import { ApiError, api } from '../../api/client'

const created = analysis({ status: 'created', video_url: null, detections: [] })
const uploaded = analysis({ status: 'uploaded', detections: [] })

function videoFile(sizeBytes = 1024, name = 'bowl.mp4'): File {
  const file = new File(['x'], name, { type: 'video/mp4' })
  Object.defineProperty(file, 'size', { value: sizeBytes })
  return file
}

function setup() {
  const onCreated = vi.fn()
  const onClose = vi.fn()
  const user = userEvent.setup()
  const { container } = render(
    <NewAnalysisModal defaultName="Session 1" onClose={onClose} onCreated={onCreated} />,
  )
  const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
  return { user, onCreated, onClose, fileInput }
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
  vi.mocked(api.limits).mockResolvedValue({ max_upload_mb: 500 })
  vi.mocked(api.createAnalysis).mockResolvedValue(created)
  vi.mocked(api.uploadVideo).mockResolvedValue(uploaded)
})

describe('NewAnalysisModal upload flow', () => {
  it('creates the analysis and uploads the chosen video, showing each stage', async () => {
    let finishUpload: (a: typeof uploaded) => void = () => undefined
    vi.mocked(api.uploadVideo).mockReturnValue(new Promise((resolve) => (finishUpload = resolve)))
    const { user, onCreated, fileInput } = setup()

    await user.upload(fileInput, videoFile(2 * 1024 * 1024))
    expect(screen.getByText('bowl.mp4')).toBeInTheDocument()
    expect(screen.getByText('2.0 MB')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))
    expect(await screen.findByRole('button', { name: /Uploading/ })).toBeDisabled()
    expect(api.createAnalysis).toHaveBeenCalledWith('Session 1')
    expect(api.uploadVideo).toHaveBeenCalledWith('a1', expect.any(File))

    finishUpload(uploaded)
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(uploaded))
  })

  it('creates an analysis without a video when none is chosen', async () => {
    const { user, onCreated } = setup()
    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(created))
    expect(api.uploadVideo).not.toHaveBeenCalled()
  })

  it('asks for a name instead of calling the backend', async () => {
    const { user } = setup()
    await user.clear(screen.getByPlaceholderText('Session 1'))
    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))
    expect(screen.getByText('Please enter a name for this analysis.')).toBeInTheDocument()
    expect(api.createAnalysis).not.toHaveBeenCalled()
  })

  it('refuses a file over the limit before sending anything', async () => {
    vi.mocked(api.limits).mockResolvedValue({ max_upload_mb: 1 })
    const { user, fileInput } = setup()
    await waitFor(() => expect(api.limits).toHaveBeenCalled())
    await user.upload(fileInput, videoFile(5 * 1024 * 1024))
    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))

    expect(
      await screen.findByText(/larger than the 1 MB limit. Trim or compress it/),
    ).toBeInTheDocument()
    expect(api.createAnalysis).not.toHaveBeenCalled()
  })

  it('shows the backend\'s worded message when the server rejects the upload', async () => {
    const detail = 'We couldn\'t read that video. Use an MP4 or MOV file, then upload it again.'
    vi.mocked(api.uploadVideo).mockRejectedValue(
      new ApiError(400, 'HTTP 400 calling /upload', 'video_unreadable', detail),
    )
    const { user, onCreated, fileInput } = setup()
    await user.upload(fileInput, videoFile())
    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))

    expect(await screen.findByText(detail)).toBeInTheDocument()
    expect(onCreated).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Create Analysis' })).toBeEnabled()
  })

  it('never shows raw error text, status codes or exception messages', async () => {
    vi.mocked(api.uploadVideo).mockRejectedValue(
      new ApiError(500, 'HTTP 500 calling /upload', null, 'Traceback: sqlite3.OperationalError'),
    )
    const { user, fileInput } = setup()
    await user.upload(fileInput, videoFile())
    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))

    expect(await screen.findByText(/BallTrace had a problem handling that/)).toBeInTheDocument()
    expect(screen.queryByText(/Traceback|sqlite|500|HTTP/)).not.toBeInTheDocument()
  })

  it('tells the user to start the backend when it cannot be reached', async () => {
    vi.mocked(api.createAnalysis).mockRejectedValue(
      new ApiError(0, 'network error: TypeError: Failed to fetch'),
    )
    const { user } = setup()
    await user.click(screen.getByRole('button', { name: 'Create Analysis' }))
    expect(await screen.findByText(/Can't reach the BallTrace backend/)).toBeInTheDocument()
    expect(screen.queryByText(/Failed to fetch|TypeError/)).not.toBeInTheDocument()
  })
})
