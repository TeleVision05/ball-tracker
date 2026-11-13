import { act, render, renderHook, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { useVideoPlayer } from './useVideoPlayer'

describe('useVideoPlayer', () => {
  it('attaches listeners when the <video> mounts after the hook (the upload flow)', () => {
    const { result } = renderHook(() => useVideoPlayer(30))
    expect(result.current.isPlaying).toBe(false)

    // The <video> element only exists after upload, so it mounts later than the hook.
    const video = document.createElement('video')
    act(() => result.current.attachVideo(video))

    act(() => {
      video.dispatchEvent(new Event('play'))
    })
    expect(result.current.isPlaying).toBe(true)

    act(() => {
      video.dispatchEvent(new Event('pause'))
    })
    expect(result.current.isPlaying).toBe(false)
  })

  it('reports duration and readiness once metadata loads', () => {
    const { result } = renderHook(() => useVideoPlayer(30))
    const video = document.createElement('video')
    Object.defineProperty(video, 'duration', { value: 12.5, configurable: true })
    act(() => result.current.attachVideo(video))

    expect(result.current.isReady).toBe(false)
    act(() => {
      video.dispatchEvent(new Event('loadedmetadata'))
    })
    expect(result.current.isReady).toBe(true)
    expect(result.current.duration).toBe(12.5)
  })

  it('follows currentTime from timeupdate and seeks within the clip', () => {
    const { result } = renderHook(() => useVideoPlayer(30))
    const video = document.createElement('video')
    Object.defineProperty(video, 'duration', { value: 10, configurable: true })
    act(() => result.current.attachVideo(video))

    video.currentTime = 3
    act(() => {
      video.dispatchEvent(new Event('timeupdate'))
    })
    expect(result.current.currentTime).toBe(3)

    act(() => result.current.seek(99))
    expect(video.currentTime).toBe(10)
    act(() => result.current.seek(-5))
    expect(video.currentTime).toBe(0)
  })

  it('stops listening when the element is detached', () => {
    const { result } = renderHook(() => useVideoPlayer(30))
    const video = document.createElement('video')
    act(() => result.current.attachVideo(video))
    act(() => result.current.attachVideo(null))
    act(() => {
      video.dispatchEvent(new Event('play'))
    })
    expect(result.current.isPlaying).toBe(false)
  })

  it('works through a real ref callback on a rendered <video>', () => {
    function Harness() {
      const { attachVideo, isPlaying } = useVideoPlayer(30)
      return (
        <>
          <video ref={attachVideo} data-testid="v" />
          <span>{isPlaying ? 'playing' : 'paused'}</span>
        </>
      )
    }
    render(<Harness />)
    expect(screen.getByText('paused')).toBeInTheDocument()
    act(() => {
      screen.getByTestId('v').dispatchEvent(new Event('play'))
    })
    expect(screen.getByText('playing')).toBeInTheDocument()
  })
})
