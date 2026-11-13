import { useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, Loader2, Save, Terminal } from 'lucide-react'
import { api } from '../../api/client'
import { friendlyError } from '../../lib/friendlyError'
import type { OllamaSettings } from '../../types'

type LoadState = 'loading' | 'ready' | 'error'

export function SettingsPanel() {
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [settings, setSettings] = useState<OllamaSettings | null>(null)

  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [windowSize, setWindowSize] = useState(4)
  const [windowOverlap, setWindowOverlap] = useState(0)
  const [frameStride, setFrameStride] = useState(3)

  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  const load = async () => {
    setLoadState('loading')
    setLoadError(null)
    try {
      const data = await api.getOllamaSettings()
      setSettings(data)
      setBaseUrl(data.base_url)
      setModel(data.model)
      setWindowSize(data.window_size)
      setWindowOverlap(data.window_overlap)
      setFrameStride(data.frame_stride ?? 3)
      setLoadState('ready')
    } catch (err) {
      setLoadError(friendlyError(err, 'load settings'))
      setLoadState('error')
    }
  }

  useEffect(() => {
    load()
  }, [])

  const handleSave = async () => {
    setSaving(true)
    setSaveError(null)
    setSaved(false)
    try {
      const updated = await api.updateOllamaSettings({
        base_url: baseUrl,
        model,
        window_size: windowSize,
        window_overlap: windowOverlap,
        frame_stride: frameStride,
      })
      setSettings(updated)
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch (err) {
      setSaveError(friendlyError(err, 'save settings'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto w-full max-w-2xl overflow-y-auto p-8">
      <h1 className="font-display text-2xl font-bold text-white">Settings</h1>
      <p className="mt-1 text-sm text-white/40">
        Configure the local Ollama vision model used for ball detection.
      </p>

      {loadState === 'loading' && (
        <div className="mt-8 flex items-center gap-2 text-sm text-white/40">
          <Loader2 size={16} className="animate-spin" />
          Loading settings…
        </div>
      )}

      {loadState === 'error' && (
        <div className="mt-8 flex items-start gap-2 rounded-md bg-red-500/10 px-3 py-2.5 text-sm text-red-300">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <p>{loadError}</p>
            <button onClick={load} className="btn-secondary mt-2">
              Retry
            </button>
          </div>
        </div>
      )}

      {loadState === 'ready' && settings && (
        <div className="mt-8 flex flex-col gap-6">
          {!settings.available && (
            <div className="flex items-start gap-3 rounded-lg border border-amber-500/25 bg-amber-500/10 px-4 py-3.5 text-sm text-amber-200">
              <AlertTriangle size={18} className="mt-0.5 shrink-0" />
              <div>
                <p className="font-medium">
                  {settings.error ?? `Ollama is not ready at ${settings.base_url}.`}
                </p>
                <p className="mt-1 text-amber-200/80">
                  Install Ollama, pull <code className="font-mono">{settings.model}</code>, and
                  ensure it&apos;s running before starting an analysis.
                </p>
                <div className="mt-2.5 flex items-center gap-2 rounded-md bg-black/30 px-3 py-2 font-mono text-xs text-white/60">
                  <Terminal size={13} className="shrink-0 text-white/40" />
                  ollama pull {settings.model} &amp;&amp; ollama serve
                </div>
              </div>
            </div>
          )}

          {settings.available && (
            <div className="flex items-center gap-2 rounded-lg border border-neon-cyan/20 bg-neon-cyan/5 px-4 py-3 text-sm text-neon-cyan">
              <CheckCircle2 size={16} />
              Connected to Ollama at {settings.base_url}
            </div>
          )}

          <div className="glass-panel flex flex-col gap-5 rounded-xl p-5">
            <div>
              <label className="label-text mb-1.5 block">Ollama Base URL</label>
              <input
                className="input-field"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://localhost:11434"
              />
            </div>

            <div>
              <label className="label-text mb-1.5 block">Model Name</label>
              <input
                className="input-field font-mono"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="qwen3-vl:8b"
              />
            </div>

            <div>
              <label className="label-text mb-1.5 block">Frame Stride</label>
              <input
                type="number"
                min={1}
                max={30}
                className="input-field"
                value={frameStride}
                onChange={(e) => setFrameStride(Number(e.target.value))}
              />
              <p className="mt-1 text-[11px] text-white/30">
                Analyze every Nth frame (3 ≈ 3× faster). Use 5–8 for quick tests; 1 for densest
                tracking.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label-text mb-1.5 block">Window Size</label>
                <input
                  type="number"
                  min={1}
                  className="input-field"
                  value={windowSize}
                  onChange={(e) => setWindowSize(Number(e.target.value))}
                />
                <p className="mt-1 text-[11px] text-white/30">
                  Used by contact-sheet mode only (advanced).
                </p>
              </div>
              <div>
                <label className="label-text mb-1.5 block">Window Overlap</label>
                <input
                  type="number"
                  min={0}
                  className="input-field"
                  value={windowOverlap}
                  onChange={(e) => setWindowOverlap(Number(e.target.value))}
                />
                <p className="mt-1 text-[11px] text-white/30">
                  Used by contact-sheet mode only (advanced).
                </p>
              </div>
            </div>

            {saveError && <p className="text-sm text-red-400">{saveError}</p>}

            <div className="flex items-center gap-3">
              <button onClick={handleSave} disabled={saving} className="btn-primary">
                {saving ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                {saving ? 'Saving…' : 'Save Settings'}
              </button>
              {saved && (
                <span className="flex items-center gap-1.5 text-sm text-neon-cyan">
                  <CheckCircle2 size={14} />
                  Saved
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
