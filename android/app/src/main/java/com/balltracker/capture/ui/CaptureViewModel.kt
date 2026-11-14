package com.balltracker.capture.ui

import android.app.Application
import android.view.TextureView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.balltracker.capture.camera.CameraController
import com.balltracker.capture.camera.CameraPhase
import com.balltracker.capture.camera.FinishedClip
import com.balltracker.capture.data.BallRecord
import com.balltracker.capture.data.CaptureStore
import com.balltracker.capture.data.RecordingOptions
import com.balltracker.capture.data.RecordingPrefs
import com.balltracker.capture.data.SavedBall
import com.balltracker.capture.data.clipLimitSeconds
import com.balltracker.capture.geometry.StumpMarks
import com.balltracker.capture.geometry.defaultStumpMarks
import com.balltracker.capture.geometry.medianInt
import com.balltracker.capture.geometry.medianLong
import java.time.OffsetDateTime
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class CaptureUiState(
    val permissionGranted: Boolean = false,
    val phase: CameraPhase = CameraPhase.Opening,
    val marks: StumpMarks? = null,
    val adjustingMarks: Boolean = true,
    val draft: StumpMarks = defaultStumpMarks(),
    val balls: List<SavedBall> = emptyList(),
    val elapsedMs: Long = 0,
    val busy: Boolean = false,
    val place: Place = Place.Film,
    val options: RecordingOptions = RecordingOptions(),
)

class CaptureViewModel(app: Application) : AndroidViewModel(app) {
    private val store = CaptureStore(app)
    private val prefs = RecordingPrefs(app)
    private val camera = CameraController(app, ::onPhase, ::onClip)
    private val _state = MutableStateFlow(initialState(store, prefs.load()))
    val state: StateFlow<CaptureUiState> = _state.asStateFlow()
    private var ticker: Job? = null

    fun onPermission(granted: Boolean) {
        _state.update { it.copy(permissionGranted = granted) }
    }

    fun attachPreview(view: TextureView) {
        if (!_state.value.permissionGranted) return
        camera.attach(view)
    }

    fun detachPreview(view: TextureView) {
        camera.detach(view)
    }

    fun nudgeMarker(which: MarkerId, dx: Double, dy: Double) {
        _state.update { current ->
            val source = current.marks ?: current.draft
            val point = source.point(which)
            val moved = source.move(
                which,
                com.balltracker.capture.geometry.clampMarker(point.x + dx, point.y + dy),
            )
            current.copy(marks = moved, draft = moved, adjustingMarks = false)
        }
    }

    fun commitMarks() {
        val marks = _state.value.marks ?: return
        val frame = camera.freezeFrame()
        if (frame != null) {
            store.saveSetup(marks, frame)
            if (!frame.isRecycled) frame.recycle()
        } else {
            store.saveMarks(marks)
        }
    }

    fun refreshPreview() {
        camera.refreshPreviewTransform()
    }

    fun open(place: Place) {
        if (_state.value.phase is CameraPhase.Recording) return
        _state.update { it.copy(place = place) }
    }

    fun deleteBall(id: String) {
        val ball = _state.value.balls.find { it.id == id } ?: return
        store.deleteBall(ball)
        _state.update {
            val next = if (it.place is Place.Ball && it.place.id == id) Place.Balls else it.place
            it.copy(balls = store.listBalls(), place = next)
        }
    }

    fun setClipSeconds(seconds: Int) {
        val options = _state.value.options.copy(maxClipSeconds = clipLimitSeconds(seconds))
        prefs.save(options)
        _state.update { it.copy(options = options) }
    }

    fun setKeepPreview(keep: Boolean) {
        val options = _state.value.options.copy(keepPreview = keep)
        prefs.save(options)
        _state.update { it.copy(options = options) }
    }

    fun toggleRecord() {
        val current = _state.value
        when (current.phase) {
            is CameraPhase.Recording -> camera.stopRecording()
            is CameraPhase.Previewing -> {
                if (current.marks == null || current.busy) return
                val (id, file) = store.nextVideoFile()
                _state.update { it.copy(busy = true) }
                camera.startRecording(file, keepPreview = current.options.keepPreview)
                pendingId = id
            }
            else -> Unit
        }
    }

    private var pendingId: String? = null

    private fun onPhase(phase: CameraPhase) {
        if (phase is CameraPhase.Recording) {
            startTicker()
        } else {
            stopTicker()
        }
        _state.update {
            it.copy(phase = phase, busy = phase is CameraPhase.Opening)
        }
    }

    private fun onClip(clip: FinishedClip) {
        val marks = _state.value.marks ?: return
        val id = pendingId ?: clip.file.nameWithoutExtension
        pendingId = null
        store.writeSidecar(
            BallRecord(
                id = id,
                recordedAt = OffsetDateTime.now().toString(),
                videoFile = clip.file.name,
                setupFrameFile = if (store.setupFrame.exists()) store.setupFrame.name else null,
                encodedWidth = clip.width,
                encodedHeight = clip.height,
                captureFps = clip.captureFps,
                fpsRangeLower = clip.fpsLower,
                fpsRangeUpper = clip.fpsUpper,
                fpsFixed = clip.fpsLower == clip.fpsUpper,
                measuredFps = clip.measuredFps,
                containerFps = clip.containerFps,
                orientationHintDegrees = clip.orientationHintDegrees,
                cameraId = clip.cameraId,
                marks = marks,
                wallClockSeconds = clip.wallClockSeconds,
                frameCount = clip.clock.frameCount,
                durationSeconds = clip.clock.durationSeconds,
                maxGapSeconds = clip.clock.maxGapSeconds,
                longGapCount = clip.clock.longGapCount,
                timestampsUs = clip.clock.timestampsUs,
                encoder = clip.encoder,
                bitrate = clip.bitrate,
                uprightWidth = clip.uprightWidth,
                uprightHeight = clip.uprightHeight,
                intrinsicSource = clip.intrinsicSource,
                intrinsicMatrixBuffer = clip.intrinsicMatrixBuffer,
                intrinsicMatrixUpright = clip.intrinsicMatrixUpright,
                focalLengthMm = clip.focalLengthMm,
                sensorWidthMm = clip.sensorWidthMm,
                sensorHeightMm = clip.sensorHeightMm,
                lensIntrinsicCalibration = clip.lensIntrinsicCalibration,
                lensDistortion = clip.lensDistortion,
                activeArray = clip.activeArray,
                preCorrectionActiveArray = clip.preCorrectionActiveArray,
                cropRegion = clip.cropRegion,
                cropChanged = clip.cropChanged,
                focusDistanceDiopters = clip.focusDistanceDiopters,
                exposureSampleCount = clip.exposureNs.size,
                exposureNsMin = clip.exposureNs.minOrNull(),
                exposureNsMedian = medianLong(clip.exposureNs),
                exposureNsMax = clip.exposureNs.maxOrNull(),
                isoMin = clip.iso.minOrNull(),
                isoMedian = medianInt(clip.iso),
                isoMax = clip.iso.maxOrNull(),
            ),
        )
        _state.update { it.copy(balls = store.listBalls(), busy = false) }
    }

    private fun startTicker() {
        ticker?.cancel()
        val started = android.os.SystemClock.elapsedRealtime()
        ticker = viewModelScope.launch {
            while (isActive) {
                val elapsed = android.os.SystemClock.elapsedRealtime() - started
                _state.update { it.copy(elapsedMs = elapsed) }
                if (elapsed >= _state.value.options.maxClipSeconds * 1000L) {
                    camera.stopRecording()
                    break
                }
                delay(200)
            }
        }
    }

    private fun stopTicker() {
        ticker?.cancel()
        ticker = null
        _state.update { it.copy(elapsedMs = 0) }
    }

    override fun onCleared() {
        ticker?.cancel()
        camera.shutdown()
        super.onCleared()
    }

    companion object {
        private fun initialState(store: CaptureStore, options: RecordingOptions): CaptureUiState {
            val marks = store.loadMarks() ?: defaultStumpMarks()
            return CaptureUiState(
                marks = marks,
                adjustingMarks = false,
                draft = marks,
                balls = store.listBalls(),
                options = options,
            )
        }
    }
}

enum class MarkerId { NearLeft, NearRight, FarLeft, FarRight }

private fun StumpMarks.point(which: MarkerId): com.balltracker.capture.geometry.NormPoint {
    return when (which) {
        MarkerId.NearLeft -> nearLeft
        MarkerId.NearRight -> nearRight
        MarkerId.FarLeft -> farLeft
        MarkerId.FarRight -> farRight
    }
}

private fun StumpMarks.move(which: MarkerId, point: com.balltracker.capture.geometry.NormPoint): StumpMarks {
    return when (which) {
        MarkerId.NearLeft -> copy(nearLeft = point)
        MarkerId.NearRight -> copy(nearRight = point)
        MarkerId.FarLeft -> copy(farLeft = point)
        MarkerId.FarRight -> copy(farRight = point)
    }
}
