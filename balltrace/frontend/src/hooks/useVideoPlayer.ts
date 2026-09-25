import { useCallback, useEffect, useRef, useState } from 'react'
import type { PlaybackSpeed } from '../types'

export interface UseVideoPlayerResult {
  videoRef: React.RefObject<HTMLVideoElement | null>
  isPlaying: boolean
  currentTime: number
  duration: number
  speed: PlaybackSpeed
  isReady: boolean
  isBuffering: boolean
  play: () => void
  pause: () => void
  togglePlay: () => void
  seek: (time: number) => void
  stepFrame: (direction: 1 | -1) => void
  setSpeed: (speed: PlaybackSpeed) => void
}

const SPEED_OPTIONS: PlaybackSpeed[] = [0.25, 0.5, 1, 1.5, 2]
export { SPEED_OPTIONS }

export function useVideoPlayer(fps: number | null): UseVideoPlayerResult {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [speed, setSpeedState] = useState<PlaybackSpeed>(1)
  const [isReady, setIsReady] = useState(false)
  const [isBuffering, setIsBuffering] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleTimeUpdate = () => setCurrentTime(video.currentTime)
    const handleLoadedMetadata = () => {
      setDuration(video.duration || 0)
      setIsReady(true)
    }
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleEnded = () => setIsPlaying(false)
    const handleWaiting = () => setIsBuffering(true)
    const handlePlaying = () => setIsBuffering(false)

    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('ended', handleEnded)
    video.addEventListener('waiting', handleWaiting)
    video.addEventListener('playing', handlePlaying)

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('ended', handleEnded)
      video.removeEventListener('waiting', handleWaiting)
      video.removeEventListener('playing', handlePlaying)
    }
  }, [videoRef.current])

  const play = useCallback(() => {
    videoRef.current?.play().catch(() => undefined)
  }, [])

  const pause = useCallback(() => {
    videoRef.current?.pause()
  }, [])

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play().catch(() => undefined)
    } else {
      video.pause()
    }
  }, [])

  const seek = useCallback((time: number) => {
    const video = videoRef.current
    if (!video) return
    const clamped = Math.max(0, Math.min(time, video.duration || time))
    video.currentTime = clamped
    setCurrentTime(clamped)
  }, [])

  const stepFrame = useCallback(
    (direction: 1 | -1) => {
      const video = videoRef.current
      if (!video) return
      const frameDuration = 1 / (fps && fps > 0 ? fps : 30)
      video.pause()
      seek(video.currentTime + direction * frameDuration)
    },
    [fps, seek],
  )

  const setSpeed = useCallback((next: PlaybackSpeed) => {
    setSpeedState(next)
    if (videoRef.current) {
      videoRef.current.playbackRate = next
    }
  }, [])

  return {
    videoRef,
    isPlaying,
    currentTime,
    duration,
    speed,
    isReady,
    isBuffering,
    play,
    pause,
    togglePlay,
    seek,
    stepFrame,
    setSpeed,
  }
}
