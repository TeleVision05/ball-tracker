import { useRef, useState } from 'react'
import { AlertTriangle, Film, Loader2, Play, UploadCloud, XCircle } from 'lucide-react'
import { api } from '../../api/client'
import { friendlyError } from '../../lib/friendlyError'
import { checkUploadSize, useUploadLimit } from '../../hooks/useUploadLimit'
import type { Analysis } from '../../types'

interface AnalysisSetupPanelProps {
  analysis: Analysis
  onUploaded: (analysis: Analysis) => void
  onStarted: () => void
}

export function AnalysisSetupPanel({ analysis, onUploaded, onStarted }: AnalysisSetupPanelProps) {
  const [uploading, setUploading] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const limitMb = useUploadLimit()

  const handleFile = async (file: File) => {
    const tooLarge = checkUploadSize(file, limitMb)
    if (tooLarge) {
      setError(tooLarge)
      return
    }
    setError(null)
    setUploading(true)
    try {
      const updated = await api.uploadVideo(analysis.id, file)
      onUploaded(updated)
    } catch (err) {
      setError(friendlyError(err, 'upload video'))
    } finally {
      setUploading(false)
    }
  }

  const handleStart = async () => {
    setError(null)
    setStarting(true)
    try {
      await api.startAnalysis(analysis.id)
      onStarted()
    } catch (err) {
      setError(friendlyError(err, 'start analysis'))
    } finally {
      setStarting(false)
    }
  }

  const needsVideo = !analysis.video_url
  const isTerminalError = analysis.status === 'failed' || analysis.status === 'cancelled'

  return (
    <div className="mx-auto flex w-full max-w-lg flex-col items-center gap-5 px-6 text-center">
      {needsVideo ? (
        <>
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white/[0.04] text-white/40">
            <UploadCloud size={26} />
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold text-white">Upload a video</h3>
            <p className="mt-1 text-sm text-white/40">
              Upload footage to analyze for ball detections.
            </p>
          </div>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              const dropped = e.dataTransfer.files?.[0]
              if (dropped) handleFile(dropped)
            }}
            onClick={() => fileInputRef.current?.click()}
            className="flex w-full cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-white/15 px-6 py-10 transition-colors hover:border-white/25 hover:bg-white/[0.02]"
          >
            {uploading ? (
              <Loader2 size={24} className="animate-spin text-neon-cyan" />
            ) : (
              <Film size={24} className="text-white/30" />
            )}
            <span className="text-sm text-white/50">
              {uploading ? 'Uploading…' : 'Click or drag a video file here'}
            </span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) handleFile(f)
            }}
          />
        </>
      ) : (
        <>
          <div
            className={`flex h-14 w-14 items-center justify-center rounded-full ${
              isTerminalError ? 'bg-red-500/10 text-red-400' : 'bg-neon-cyan/10 text-neon-cyan'
            }`}
          >
            {isTerminalError ? <XCircle size={26} /> : <Play size={26} />}
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold text-white">
              {isTerminalError ? 'Analysis did not complete' : 'Ready to analyze'}
            </h3>
            <p className="mt-1 text-sm text-white/40">
              {isTerminalError
                ? analysis.error ?? `Analysis was ${analysis.status}.`
                : `${analysis.frame_count ?? '?'} frames · ${analysis.fps?.toFixed(1) ?? '?'} fps · ${
                    analysis.duration?.toFixed(1) ?? '?'
                  }s`}
            </p>
          </div>
          <button onClick={handleStart} disabled={starting} className="btn-primary shadow-glow-sm">
            {starting ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
            {starting ? 'Starting…' : isTerminalError ? 'Retry Analysis' : 'Start Analysis'}
          </button>
        </>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-md bg-red-500/10 px-3 py-2 text-sm text-red-300">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}
    </div>
  )
}
