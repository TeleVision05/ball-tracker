import { useRef, useState } from 'react'
import { X, UploadCloud, Loader2, Film, AlertTriangle } from 'lucide-react'
import { api } from '../../api/client'
import { friendlyError } from '../../lib/friendlyError'
import { checkUploadSize, useUploadLimit } from '../../hooks/useUploadLimit'
import type { Analysis } from '../../types'

interface NewAnalysisModalProps {
  defaultName: string
  onClose: () => void
  onCreated: (analysis: Analysis) => void
}

type Stage = 'form' | 'creating' | 'uploading' | 'error'

export function NewAnalysisModal({ defaultName, onClose, onCreated }: NewAnalysisModalProps) {
  const [name, setName] = useState(defaultName)
  const [file, setFile] = useState<File | null>(null)
  const [stage, setStage] = useState<Stage>('form')
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const limitMb = useUploadLimit()

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const dropped = e.dataTransfer.files?.[0]
    if (dropped) setFile(dropped)
  }

  const handleSubmit = async () => {
    if (!name.trim()) {
      setError('Please enter a name for this analysis.')
      return
    }
    const tooLarge = file ? checkUploadSize(file, limitMb) : null
    if (tooLarge) {
      setError(tooLarge)
      return
    }
    setError(null)
    setStage('creating')
    try {
      const created = await api.createAnalysis(name.trim())
      if (!file) {
        onCreated(created)
        return
      }
      setStage('uploading')
      const uploaded = await api.uploadVideo(created.id, file)
      onCreated(uploaded)
    } catch (err) {
      setError(friendlyError(err, file ? 'upload video' : 'create analysis'))
      setStage('error')
    }
  }

  const busy = stage === 'creating' || stage === 'uploading'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="glass-surface w-full max-w-md rounded-xl p-6 shadow-2xl">
        <div className="flex items-center justify-between">
          <h2 className="font-display text-lg font-semibold text-white">New Analysis</h2>
          <button onClick={onClose} className="btn-ghost" disabled={busy}>
            <X size={18} />
          </button>
        </div>

        <div className="mt-5 flex flex-col gap-4">
          <div>
            <label className="label-text mb-1.5 block">Name</label>
            <input
              autoFocus
              className="input-field"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Session 1"
              disabled={busy}
            />
          </div>

          <div>
            <label className="label-text mb-1.5 block">Video (optional now, upload later)</label>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-6 text-center transition-colors ${
                file
                  ? 'border-neon-cyan/40 bg-neon-cyan/5'
                  : 'border-white/15 hover:border-white/25 hover:bg-white/[0.02]'
              }`}
            >
              {file ? (
                <>
                  <Film size={22} className="text-neon-cyan" />
                  <span className="max-w-full truncate text-sm text-white/80">{file.name}</span>
                  <span className="text-xs text-white/35">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </span>
                </>
              ) : (
                <>
                  <UploadCloud size={22} className="text-white/30" />
                  <span className="text-sm text-white/50">Click or drag a video file here</span>
                </>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              disabled={busy}
            />
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md bg-red-500/10 px-3 py-2 text-sm text-red-300">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          <div className="mt-1 flex justify-end gap-2">
            <button onClick={onClose} className="btn-secondary" disabled={busy}>
              Cancel
            </button>
            <button onClick={handleSubmit} className="btn-primary" disabled={busy}>
              {busy && <Loader2 size={15} className="animate-spin" />}
              {stage === 'creating' && 'Creating…'}
              {stage === 'uploading' && 'Uploading…'}
              {!busy && 'Create Analysis'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
