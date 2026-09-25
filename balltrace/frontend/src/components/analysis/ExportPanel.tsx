import { useState } from 'react'
import { FileJson, FileSpreadsheet, Film, Loader2, Download } from 'lucide-react'
import { api, ApiError } from '../../api/client'
import type { TrajectorySettings } from '../../types'

interface ExportPanelProps {
  analysisId: string
  trajectorySettings: TrajectorySettings
}

function triggerDownload(url: string) {
  const link = document.createElement('a')
  link.href = url
  link.target = '_blank'
  link.rel = 'noopener noreferrer'
  document.body.appendChild(link)
  link.click()
  link.remove()
}

export function ExportPanel({ analysisId, trajectorySettings }: ExportPanelProps) {
  const [videoState, setVideoState] = useState<'idle' | 'rendering' | 'error'>('idle')
  const [videoError, setVideoError] = useState<string | null>(null)

  const handleVideoExport = async () => {
    setVideoState('rendering')
    setVideoError(null)
    try {
      const result = await api.exportVideo(analysisId, {
        line_thickness: trajectorySettings.lineThickness,
        max_gap_frames: trajectorySettings.maxGapFrames,
        show_markers: trajectorySettings.showMarkers,
      })
      const downloadUrl = result.download_url ?? api.exportVideoFileUrl(analysisId)
      triggerDownload(downloadUrl)
      setVideoState('idle')
    } catch (err) {
      setVideoState('error')
      setVideoError(err instanceof ApiError ? err.message : 'Video export failed.')
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={() => triggerDownload(api.exportJsonUrl(analysisId))}
        className="btn-secondary w-full justify-start"
      >
        <FileJson size={15} />
        Export JSON
        <Download size={13} className="ml-auto text-white/30" />
      </button>
      <button
        onClick={() => triggerDownload(api.exportCsvUrl(analysisId))}
        className="btn-secondary w-full justify-start"
      >
        <FileSpreadsheet size={15} />
        Export CSV
        <Download size={13} className="ml-auto text-white/30" />
      </button>
      <button
        onClick={handleVideoExport}
        disabled={videoState === 'rendering'}
        className="btn-secondary w-full justify-start"
      >
        {videoState === 'rendering' ? (
          <Loader2 size={15} className="animate-spin" />
        ) : (
          <Film size={15} />
        )}
        {videoState === 'rendering' ? 'Rendering video…' : 'Export Video (burned-in)'}
        {videoState !== 'rendering' && <Download size={13} className="ml-auto text-white/30" />}
      </button>
      {videoState === 'error' && videoError && (
        <p className="text-xs text-red-400">{videoError}</p>
      )}
    </div>
  )
}
