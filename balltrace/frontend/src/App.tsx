import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, Loader2, Radar } from 'lucide-react'
import { AppShell } from './components/layout/AppShell'
import { LeftSidebar } from './components/layout/LeftSidebar'
import { RightSidebar } from './components/layout/RightSidebar'
import { VideoPlayer, type VideoPlayerHandle } from './components/player/VideoPlayer'
import { ProgressPanel } from './components/analysis/ProgressPanel'
import { AnalysisSetupPanel } from './components/analysis/AnalysisSetupPanel'
import { NewAnalysisModal } from './components/analysis/NewAnalysisModal'
import { SettingsPanel } from './components/settings/SettingsPanel'
import { useAnalysisProgress } from './hooks/useAnalysisProgress'
import { api } from './api/client'
import { friendlyError } from './lib/friendlyError'
import type { Analysis, AnalysisSummary, Detection, TrajectorySettings } from './types'

type View = 'analysis' | 'settings'

const DEFAULT_TRAJECTORY_SETTINGS: TrajectorySettings = {
  showTrajectory: true,
  showMarkers: true,
  showConfidence: false,
  lineThickness: 3,
  maxGapFrames: 10,
}

function EmptyState({ onNewAnalysis }: { onNewAnalysis: () => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-neon-cyan/[0.06] text-neon-cyan shadow-glow-sm">
        <Radar size={30} strokeWidth={1.75} />
      </div>
      <div>
        <h2 className="font-display text-xl font-semibold text-white">No analysis selected</h2>
        <p className="mt-1.5 max-w-sm text-sm text-white/40">
          Create a new analysis and upload cricket footage to start tracking ball trajectory.
        </p>
      </div>
      <button onClick={onNewAnalysis} className="btn-primary mt-2 shadow-glow-sm">
        New Analysis
      </button>
    </div>
  )
}

export default function App() {
  const [view, setView] = useState<View>('analysis')

  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([])
  const [analysesLoading, setAnalysesLoading] = useState(true)
  const [analysesError, setAnalysesError] = useState<string | null>(null)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedAnalysis, setSelectedAnalysis] = useState<Analysis | null>(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)

  const [showNewModal, setShowNewModal] = useState(false)
  const [trajectorySettings, setTrajectorySettings] = useState<TrajectorySettings>(
    DEFAULT_TRAJECTORY_SETTINGS,
  )
  const [cancelling, setCancelling] = useState(false)

  const [currentTime, setCurrentTime] = useState(0)
  const [currentDetection, setCurrentDetection] = useState<Detection | null>(null)
  const videoPlayerRef = useRef<VideoPlayerHandle>(null)

  const refreshAnalysesList = useCallback(async () => {
    setAnalysesLoading(true)
    setAnalysesError(null)
    try {
      const data = await api.listAnalyses()
      setAnalyses(data.sort((a, b) => (a.created_at < b.created_at ? 1 : -1)))
    } catch (err) {
      setAnalysesError(friendlyError(err, 'load analyses'))
    } finally {
      setAnalysesLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshAnalysesList()
  }, [refreshAnalysesList])

  const loadAnalysis = useCallback(async (id: string) => {
    setAnalysisLoading(true)
    setAnalysisError(null)
    try {
      const data = await api.getAnalysis(id)
      setSelectedAnalysis(data)
    } catch (err) {
      setAnalysisError(friendlyError(err, 'load analysis'))
    } finally {
      setAnalysisLoading(false)
    }
  }, [])

  const handleSelectAnalysis = (id: string) => {
    setView('analysis')
    setSelectedId(id)
    setCurrentTime(0)
    setCurrentDetection(null)
    loadAnalysis(id)
  }

  const handleCreated = (analysis: Analysis) => {
    setShowNewModal(false)
    setSelectedAnalysis(analysis)
    setSelectedId(analysis.id)
    setView('analysis')
    refreshAnalysesList()
  }

  const handleUploaded = (analysis: Analysis) => {
    setSelectedAnalysis(analysis)
    refreshAnalysesList()
  }

  const handleStarted = () => {
    if (!selectedAnalysis) return
    setSelectedAnalysis({ ...selectedAnalysis, status: 'analyzing', error: null })
    refreshAnalysesList()
  }

  const handleProgressTerminal = useCallback(() => {
    if (selectedId) loadAnalysis(selectedId)
    refreshAnalysesList()
  }, [selectedId, loadAnalysis, refreshAnalysesList])

  const { progress, connected, streamError } = useAnalysisProgress(
    selectedId,
    selectedAnalysis?.status === 'analyzing',
    handleProgressTerminal,
  )

  const handleCancel = async () => {
    if (!selectedId) return
    setCancelling(true)
    try {
      await api.cancelAnalysis(selectedId)
    } catch {
      // Cancellation errors surface via the next progress/status refresh.
    } finally {
      setCancelling(false)
    }
  }

  const handleClear = () => {
    setSelectedId(null)
    setSelectedAnalysis(null)
    setAnalysisError(null)
  }

  const handleRerun = async () => {
    if (!selectedAnalysis) return
    try {
      await api.startAnalysis(selectedAnalysis.id)
      handleStarted()
    } catch (err) {
      setAnalysisError(friendlyError(err, 'start analysis'))
    }
  }

  const handleVideoTimeUpdate = useCallback((time: number, detection: Detection | null) => {
    setCurrentTime(time)
    setCurrentDetection(detection)
  }, [])

  const handleSeekFromSidebar = (time: number) => {
    videoPlayerRef.current?.seek(time)
  }

  let mainContent: React.ReactNode
  if (view === 'settings') {
    mainContent = <SettingsPanel />
  } else if (!selectedId) {
    mainContent = <EmptyState onNewAnalysis={() => setShowNewModal(true)} />
  } else if (analysisLoading) {
    mainContent = (
      <div className="flex h-full items-center justify-center text-white/40">
        <Loader2 size={24} className="animate-spin" />
      </div>
    )
  } else if (analysisError) {
    mainContent = (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <AlertTriangle size={24} className="text-red-400" />
        <p className="max-w-sm text-sm text-white/50">{analysisError}</p>
        <button onClick={() => loadAnalysis(selectedId)} className="btn-secondary">
          Retry
        </button>
      </div>
    )
  } else if (selectedAnalysis) {
    if (selectedAnalysis.status === 'analyzing') {
      mainContent = (
        <div className="flex h-full items-center justify-center px-6">
          <ProgressPanel
            progress={progress}
            connected={connected}
            streamError={streamError}
            onCancel={handleCancel}
            cancelling={cancelling}
          />
        </div>
      )
    } else if (selectedAnalysis.status === 'completed') {
      mainContent = (
        <VideoPlayer
          ref={videoPlayerRef}
          analysis={selectedAnalysis}
          detections={selectedAnalysis.detections}
          trajectorySettings={trajectorySettings}
          onTimeUpdate={handleVideoTimeUpdate}
        />
      )
    } else {
      mainContent = (
        <div className="flex h-full items-center justify-center px-6">
          <AnalysisSetupPanel
            analysis={selectedAnalysis}
            onUploaded={handleUploaded}
            onStarted={handleStarted}
          />
        </div>
      )
    }
  }

  const showRightSidebar = view === 'analysis' && Boolean(selectedAnalysis)

  return (
    <>
      <AppShell
        sidebar={
          <LeftSidebar
            analyses={analyses}
            analysesLoading={analysesLoading}
            analysesError={analysesError}
            selectedId={selectedId}
            activeView={view}
            onSelectAnalysis={handleSelectAnalysis}
            onNewAnalysis={() => setShowNewModal(true)}
            onOpenSettings={() => setView('settings')}
          />
        }
        main={mainContent}
        rightPanel={
          showRightSidebar && selectedAnalysis ? (
            <RightSidebar
              analysis={selectedAnalysis}
              detections={selectedAnalysis.detections}
              currentTime={currentTime}
              currentDetection={currentDetection}
              trajectorySettings={trajectorySettings}
              onTrajectorySettingsChange={(patch) =>
                setTrajectorySettings((prev) => ({ ...prev, ...patch }))
              }
              onSeek={handleSeekFromSidebar}
              onClear={handleClear}
              onRerun={handleRerun}
            />
          ) : null
        }
      />

      {showNewModal && (
        <NewAnalysisModal
          defaultName={`Session ${analyses.length + 1}`}
          onClose={() => setShowNewModal(false)}
          onCreated={handleCreated}
        />
      )}
    </>
  )
}
