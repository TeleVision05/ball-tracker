package com.balltracker.capture.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.view.Surface
import android.view.TextureView
import com.balltracker.capture.geometry.ClipClock
import com.balltracker.capture.geometry.describeCamera
import com.balltracker.capture.geometry.displayRotationDegrees
import com.balltracker.capture.geometry.lensMatrix
import com.balltracker.capture.geometry.medianInt
import com.balltracker.capture.geometry.medianLong
import com.balltracker.capture.geometry.orientationHintDegrees
import java.io.File
import java.util.concurrent.Executor
import kotlin.math.roundToInt

data class FinishedClip(
    val file: File,
    val measuredFps: Double?,
    val containerFps: Int,
    val width: Int,
    val height: Int,
    val captureFps: Int,
    val fpsLower: Int,
    val fpsUpper: Int,
    val orientationHintDegrees: Int,
    val cameraId: String,
    val wallClockSeconds: Double,
    val clock: ClipClock,
    val encoder: String,
    val bitrate: Int,
    val uprightWidth: Int,
    val uprightHeight: Int,
    val intrinsicSource: String?,
    val intrinsicMatrixBuffer: List<List<Double>>?,
    val intrinsicMatrixUpright: List<List<Double>>?,
    val focalLengthMm: Double?,
    val sensorWidthMm: Double?,
    val sensorHeightMm: Double?,
    val lensIntrinsicCalibration: List<Double>?,
    val lensDistortion: List<Double>?,
    val activeArray: List<Int>?,
    val preCorrectionActiveArray: List<Int>?,
    val cropRegion: List<Int>?,
    val cropChanged: Boolean,
    val focusDistanceDiopters: Double?,
    val exposureNs: List<Long>,
    val iso: List<Int>,
)

sealed interface CameraPhase {
    data object Opening : CameraPhase
    data class Previewing(val profile: ChosenProfile, val note: String? = null) : CameraPhase
    data class Recording(val profile: ChosenProfile) : CameraPhase
    data class Failed(val message: String) : CameraPhase
}

/**
 * Films with the phone's high-speed camera, the same path Samsung uses for slow motion.
 * Preview and recording share one size so the stump marks land on the frames that were saved.
 */
class CameraController(
    context: Context,
    private val onPhase: (CameraPhase) -> Unit,
    private val onClip: (FinishedClip) -> Unit,
) {
    private val appContext = context.applicationContext
    private val manager = appContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val main = Handler(android.os.Looper.getMainLooper())
    private val thread = HandlerThread("balltracker-camera").also { it.start() }
    private val cameraHandler = Handler(thread.looper)
    private val cameraExecutor = Executor { command -> cameraHandler.post(command) }

    private var generation = 0
    private var preview: TextureView? = null
    private var previewSurface: Surface? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var recorder: MediaRecorder? = null
    private var profile: ChosenProfile? = null
    private var sensorOrientation = 90
    private var pendingFile: File? = null
    private var containerFps = 30
    private var encoderName = "h264"
    private var bitrate = 0
    private var recordStartedMs = 0L
    private var counting = false
    private var frameSeen = false
    private var pendingNote: String? = null
    private var highSpeedPreviewRefused = false
    private var reopenAttempts = 0
    private var recordPreviewIncluded = true
    private var lens = LensFacts()
    private var focusDiopters: Float? = null
    private var clipFocalLengthMm: Double? = null
    private var cropRegion: List<Int>? = null
    private var cropChanged = false
    private val exposureNs = ArrayList<Long>()
    private val isoSamples = ArrayList<Int>()

    fun attach(view: TextureView) {
        preview = view
        view.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
                applyPreviewTransform()
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                if (preview === view) {
                    preview = null
                    closeCamera()
                }
                return true
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                frameSeen = true
            }
        }
        if (view.isAvailable) openCamera()
    }

    fun detach(view: TextureView) {
        if (preview !== view) return
        preview = null
        closeCamera()
    }

    fun shutdown() {
        preview = null
        closeCamera()
        thread.quitSafely()
    }

    fun freezeFrame(): Bitmap? {
        val view = preview ?: return null
        val chosen = profile ?: return null
        if (!frameSeen || !view.isAvailable) return null
        val raw = view.getBitmap(chosen.width, chosen.height) ?: return null
        val hint = currentHint()
        if (hint == 0) return raw
        val matrix = Matrix().apply { postRotate(hint.toFloat()) }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
    }

    fun startRecording(file: File, keepPreview: Boolean = false) {
        cameraHandler.post {
            val chosen = profile
            val surface = previewSurface
            val camera = device
            if (chosen == null || surface == null || camera == null) {
                publish(CameraPhase.Failed("The camera is not ready yet."))
                return@post
            }
            try {
                session?.close()
                session = null
                releaseRecorder()
                val prepared = prepareRecorder(file, chosen)
                recorder = prepared
                pendingFile = file
                resetClipStats()
                recordPreviewIncluded = keepPreview || !chosen.slowMotion
                openRecordSession()
            } catch (error: Exception) {
                Log.e(TAG, "startRecording", error)
                file.delete()
                pendingFile = null
                releaseRecorder()
                pendingNote = error.message ?: "Recording did not start."
                main.post { createPreviewSession() }
            }
        }
    }

    fun stopRecording() {
        Log.i(TAG, "stop requested")
        cameraHandler.post { finishRecording() }
    }

    fun refreshPreviewTransform() {
        main.post { applyPreviewTransform() }
    }

    private fun openCamera() {
        cameraHandler.post {
            if (preview == null || device != null) return@post
            val chosen = chooseProfile(readOffers())
            if (chosen == null) {
                publish(CameraPhase.Failed("No camera on this phone can record video."))
                return@post
            }
            profile = chosen
            val characteristics = manager.getCameraCharacteristics(chosen.cameraId)
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            lens = readLens(characteristics)
            publish(CameraPhase.Opening)
            val token = generation
            try {
                openDevice(chosen.cameraId, token)
            } catch (error: SecurityException) {
                publish(CameraPhase.Failed("Camera permission is required to film."))
            } catch (error: Exception) {
                Log.e(TAG, "openCamera", error)
                publish(CameraPhase.Failed(error.message ?: "The camera did not open."))
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun openDevice(cameraId: String, token: Int) {
        manager.openCamera(
            cameraId,
            object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (token != generation) {
                        camera.close()
                        return
                    }
                    device = camera
                    reopenAttempts = 0
                    main.post { createPreviewSession() }
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    if (device === camera) device = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "camera error $error")
                    camera.close()
                    if (device === camera) device = null
                    session?.close()
                    session = null
                    if (preview != null && reopenAttempts < 3) {
                        reopenAttempts += 1
                        publish(CameraPhase.Opening)
                        cameraHandler.postDelayed({
                            if (device == null && preview != null) openCamera()
                        }, 800)
                    } else {
                        publish(CameraPhase.Failed("The camera stopped. Close the app and open it again."))
                    }
                }
            },
            cameraHandler,
        )
    }

    private fun createPreviewSession() {
        val view = preview ?: return
        val chosen = profile ?: return
        val texture = view.surfaceTexture ?: return
        texture.setDefaultBufferSize(chosen.width, chosen.height)
        applyPreviewTransform()
        previewSurface?.release()
        val surface = Surface(texture)
        previewSurface = surface
        val token = generation
        val useHighSpeed = chosen.slowMotion && !highSpeedPreviewRefused
        cameraHandler.post {
            val camera = device ?: return@post
            if (token != generation) return@post
            try {
                val targets = listOf(surface)
                if (useHighSpeed) {
                    camera.createCaptureSession(
                        SessionConfiguration(
                            SessionConfiguration.SESSION_HIGH_SPEED,
                            targets.map { OutputConfiguration(it) },
                            cameraExecutor,
                            sessionCallback(token, targets, startRecorder = false),
                        ),
                    )
                } else {
                    camera.createCaptureSession(
                        targets,
                        sessionCallback(token, targets, startRecorder = false),
                        cameraHandler,
                    )
                }
            } catch (error: Exception) {
                Log.e(TAG, "createPreviewSession", error)
                if (useHighSpeed) {
                    highSpeedPreviewRefused = true
                    main.post { createPreviewSession() }
                } else {
                    publish(CameraPhase.Failed(error.message ?: "The preview did not start."))
                }
            }
        }
    }

    private fun openRecordSession() {
        val camera = device ?: return
        val recorderSurface = recorder?.surface ?: return
        val chosen = profile ?: return
        val targets = ArrayList<Surface>()
        if (recordPreviewIncluded) previewSurface?.let { targets.add(it) }
        targets.add(recorderSurface)
        if (targets.isEmpty()) return
        val token = generation
        Log.i(TAG, "record session preview=$recordPreviewIncluded targets=${targets.size}")
        try {
            if (chosen.slowMotion) {
                camera.createCaptureSession(
                    SessionConfiguration(
                        SessionConfiguration.SESSION_HIGH_SPEED,
                        targets.map { OutputConfiguration(it) },
                        cameraExecutor,
                        sessionCallback(token, targets, startRecorder = true),
                    ),
                )
            } else {
                camera.createCaptureSession(
                    targets,
                    sessionCallback(token, targets, startRecorder = true),
                    cameraHandler,
                )
            }
        } catch (error: Exception) {
            Log.e(TAG, "openRecordSession", error)
            if (!recordPreviewIncluded) {
                recordPreviewIncluded = true
                openRecordSession()
            } else {
                pendingFile?.delete()
                pendingFile = null
                releaseRecorder()
                pendingNote = error.message ?: "Recording did not start."
                main.post { createPreviewSession() }
            }
        }
    }

    private fun retryRecordWithPreview(failed: CameraCaptureSession): Boolean {
        if (recordPreviewIncluded) return false
        recordPreviewIncluded = true
        failed.close()
        session = null
        cameraHandler.post { openRecordSession() }
        return true
    }

    private fun sessionCallback(
        token: Int,
        targets: List<Surface>,
        startRecorder: Boolean,
    ) = object : CameraCaptureSession.StateCallback() {
        override fun onConfigured(captureSession: CameraCaptureSession) {
            if (token != generation || device == null) {
                captureSession.close()
                return
            }
            session = captureSession
            try {
                val request = repeatingRequest(targets, lockFocus = startRecorder)
                if (captureSession is CameraConstrainedHighSpeedCaptureSession) {
                    captureSession.setRepeatingBurst(
                        captureSession.createHighSpeedRequestList(request),
                        frameCounter(),
                        cameraHandler,
                    )
                } else {
                    captureSession.setRepeatingRequest(request, frameCounter(), cameraHandler)
                }
                if (startRecorder) {
                    recorder?.start()
                    recordStartedMs = SystemClock.elapsedRealtime()
                    counting = true
                    profile?.let { publish(CameraPhase.Recording(it)) }
                } else {
                    profile?.let { publish(CameraPhase.Previewing(it, pendingNote)) }
                    pendingNote = null
                }
            } catch (error: Exception) {
                Log.e(TAG, "onConfigured", error)
                if (startRecorder && retryRecordWithPreview(captureSession)) return
                if (!startRecorder && profile?.slowMotion == true && !highSpeedPreviewRefused) {
                    highSpeedPreviewRefused = true
                    captureSession.close()
                    main.post { createPreviewSession() }
                } else {
                    publish(CameraPhase.Failed(error.message ?: "The camera could not keep up."))
                }
            }
        }

        override fun onConfigureFailed(captureSession: CameraCaptureSession) {
            if (startRecorder && retryRecordWithPreview(captureSession)) return
            if (!startRecorder && profile?.slowMotion == true && !highSpeedPreviewRefused) {
                highSpeedPreviewRefused = true
                main.post { createPreviewSession() }
                return
            }
            publish(
                CameraPhase.Failed(
                    if (startRecorder) {
                        "This phone refused slow-motion recording."
                    } else {
                        "This phone refused the slow-motion camera."
                    },
                ),
            )
        }
    }

    private fun repeatingRequest(targets: List<Surface>, lockFocus: Boolean): CaptureRequest {
        val chosen = profile ?: error("No profile")
        val camera = device ?: error("No camera")
        val builder = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        targets.forEach { builder.addTarget(it) }
        builder.set(
            CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
            Range(chosen.fpsLower, chosen.fpsUpper),
        )
        val distance = focusDiopters
        if (lockFocus && distance != null) {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, distance)
        } else {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        }
        builder.set(
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
        )
        builder.set(
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
        )
        return builder.build()
    }

    private fun frameCounter() = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: android.hardware.camera2.TotalCaptureResult,
        ) {
            if (!counting) {
                result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let { focusDiopters = it }
                return
            }
            result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { exposureNs.add(it) }
            result.get(CaptureResult.SENSOR_SENSITIVITY)?.let { isoSamples.add(it) }
            result.get(CaptureResult.LENS_FOCAL_LENGTH)?.let { reported ->
                if (clipFocalLengthMm == null) clipFocalLengthMm = reported.toDouble()
            }
            val region = result.get(CaptureResult.SCALER_CROP_REGION) ?: return
            val next = listOf(region.left, region.top, region.right, region.bottom)
            val current = cropRegion
            if (current == null) cropRegion = next else if (current != next) cropChanged = true
        }
    }

    private fun finishRecording() {
        counting = false
        val file = pendingFile
        val chosen = profile
        val started = recordStartedMs
        val hint = currentHint()
        val exposures = exposureNs.toList()
        val isos = isoSamples.toList()
        val crop = cropRegion
        val cropDidChange = cropChanged
        val focal = clipFocalLengthMm ?: lens.focalLengthMm
        val focus = focusDiopters?.toDouble()
        pendingFile = null
        try {
            recorder?.stop()
        } catch (error: RuntimeException) {
            Log.w(TAG, "recorder stop", error)
            file?.delete()
            teardownRecorder()
            pendingNote = "Hold the button through the delivery."
            main.post { createPreviewSession() }
            return
        }
        teardownRecorder()
        val tooShort = started == 0L || SystemClock.elapsedRealtime() - started < 400
        if (file == null || chosen == null || tooShort || !file.exists()) {
            file?.delete()
            pendingNote = "Hold the button through the delivery."
            main.post { createPreviewSession() }
            return
        }
        val wallClockSeconds = (SystemClock.elapsedRealtime() - started) / 1000.0
        val clock = com.balltracker.capture.geometry.clipClock(
            readTimestamps(file),
            chosen.captureFps,
        )
        val described = describeCamera(
            intrinsic = lens.intrinsic,
            focalLengthMm = focal,
            sensorWidthMm = lens.sensorWidthMm,
            sensorHeightMm = lens.sensorHeightMm,
            active = lens.activeArray,
            pre = lens.preCorrectionArray,
            crop = crop,
            bufferWidth = chosen.width,
            bufferHeight = chosen.height,
            clockwiseDegrees = hint,
        )
        val measured = clock.measuredFps
        pendingNote = when {
            measured == null -> "Could not read the frame times from that clip."
            measured < chosen.captureFps - 5 ->
                "Last clip is ${measured.roundToInt()} fps. The times in the file are the real clock."
            else -> "Last clip held ${measured.roundToInt()} fps."
        }
        Log.i(
            TAG,
            "clip frames=${clock.frameCount} fps=$measured maxGap=${clock.maxGapSeconds} encoder=$encoderName",
        )
        profile?.let { publish(CameraPhase.Opening) }
        main.post {
            onClip(
                FinishedClip(
                    file = file,
                    measuredFps = measured,
                    containerFps = containerFps,
                    width = chosen.width,
                    height = chosen.height,
                    captureFps = chosen.captureFps,
                    fpsLower = chosen.fpsLower,
                    fpsUpper = chosen.fpsUpper,
                    orientationHintDegrees = hint,
                    cameraId = chosen.cameraId,
                    wallClockSeconds = wallClockSeconds,
                    clock = clock,
                    encoder = encoderName,
                    bitrate = bitrate,
                    uprightWidth = described.uprightWidth,
                    uprightHeight = described.uprightHeight,
                    intrinsicSource = described.source,
                    intrinsicMatrixBuffer = described.buffer?.let { lensMatrix(it) },
                    intrinsicMatrixUpright = described.upright?.let { lensMatrix(it) },
                    focalLengthMm = focal,
                    sensorWidthMm = lens.sensorWidthMm,
                    sensorHeightMm = lens.sensorHeightMm,
                    lensIntrinsicCalibration = lens.intrinsic,
                    lensDistortion = lens.distortion,
                    activeArray = lens.activeArray,
                    preCorrectionActiveArray = lens.preCorrectionArray,
                    cropRegion = crop,
                    cropChanged = cropDidChange,
                    focusDistanceDiopters = focus,
                    exposureNs = exposures,
                    iso = isos,
                ),
            )
        }
        cameraHandler.postDelayed({ main.post { createPreviewSession() } }, 400)
    }

    private fun teardownRecorder() {
        session?.close()
        session = null
        releaseRecorder()
    }

    private fun prepareRecorder(file: File, chosen: ChosenProfile): MediaRecorder {
        val attempts = listOf(
            RecorderAttempt(chosen.captureFps, MediaRecorder.VideoEncoder.HEVC, "hevc"),
            RecorderAttempt(chosen.captureFps, MediaRecorder.VideoEncoder.H264, "h264"),
            RecorderAttempt(30, MediaRecorder.VideoEncoder.H264, "h264"),
        )
        var last: Exception? = null
        for (attempt in attempts) {
            val candidate = if (Build.VERSION.SDK_INT >= 31) {
                MediaRecorder(appContext)
            } else {
                MediaRecorder()
            }
            try {
                candidate.setVideoSource(MediaRecorder.VideoSource.SURFACE)
                candidate.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                candidate.setVideoEncoder(attempt.encoder)
                candidate.setVideoSize(chosen.width, chosen.height)
                candidate.setVideoFrameRate(attempt.containerFps)
                candidate.setCaptureRate(chosen.captureFps.toDouble())
                val rate = bitRate(chosen)
                candidate.setVideoEncodingBitRate(rate)
                candidate.setOrientationHint(currentHint())
                candidate.setOutputFile(file.absolutePath)
                candidate.prepare()
                containerFps = attempt.containerFps
                encoderName = attempt.name
                bitrate = rate
                return candidate
            } catch (error: Exception) {
                last = error
                candidate.reset()
                candidate.release()
            }
        }
        throw last ?: IllegalStateException("Recording did not start.")
    }

    private fun bitRate(chosen: ChosenProfile): Int {
        val estimate = chosen.width.toLong() * chosen.height * chosen.captureFps / 8L
        return estimate.coerceIn(12_000_000L, 100_000_000L).toInt()
    }

    private fun resetClipStats() {
        exposureNs.clear()
        isoSamples.clear()
        cropRegion = null
        cropChanged = false
        clipFocalLengthMm = null
    }

    private fun readTimestamps(file: File): List<Long> {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(file.absolutePath)
            val track = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: return emptyList()
            extractor.selectTrack(track)
            val times = ArrayList<Long>()
            while (true) {
                val time = extractor.sampleTime
                if (time < 0) break
                times.add(time)
                if (!extractor.advance()) break
            }
            times
        } catch (error: Exception) {
            Log.w(TAG, "timestamps", error)
            emptyList()
        } finally {
            extractor.release()
        }
    }

    private fun readLens(chars: CameraCharacteristics): LensFacts {
        val focal = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()?.toDouble()
        val sensor = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val intrinsic = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            ?.map { it.toDouble() }
        val distortion = chars.get(CameraCharacteristics.LENS_DISTORTION)?.map { it.toDouble() }
        return LensFacts(
            focalLengthMm = focal,
            sensorWidthMm = sensor?.width?.toDouble(),
            sensorHeightMm = sensor?.height?.toDouble(),
            intrinsic = intrinsic,
            distortion = distortion,
            activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.let {
                listOf(it.left, it.top, it.right, it.bottom)
            },
            preCorrectionArray = chars.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)?.let {
                listOf(it.left, it.top, it.right, it.bottom)
            },
        )
    }

    private fun currentHint(): Int {
        val rotation = preview?.display?.rotation ?: Surface.ROTATION_90
        return orientationHintDegrees(sensorOrientation, displayRotationDegrees(rotation))
    }

    private fun applyPreviewTransform() {
        val view = preview ?: return
        val chosen = profile ?: return
        if (view.width == 0 || view.height == 0) return
        val rotation = view.display?.rotation ?: return
        view.setTransform(
            previewMatrix(view.width, view.height, chosen.width, chosen.height, rotation),
        )
    }

    private fun closeCamera() {
        generation += 1
        publish(CameraPhase.Opening)
        cameraHandler.post {
            try {
                recorder?.stop()
            } catch (_: RuntimeException) {
            }
            releaseRecorder()
            pendingFile?.delete()
            pendingFile = null
            session?.close()
            session = null
            device?.close()
            device = null
            previewSurface?.release()
            previewSurface = null
            frameSeen = false
        }
    }

    private fun releaseRecorder() {
        try {
            recorder?.reset()
        } catch (_: Exception) {
        }
        try {
            recorder?.release()
        } catch (_: Exception) {
        }
        recorder = null
    }

    private fun readOffers(): List<CameraOffer> {
        return manager.cameraIdList.mapNotNull { id ->
            val chars = manager.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return@mapNotNull null
            val highSpeed = map.highSpeedVideoSizes?.map { size ->
                HighSpeedOption(
                    size = VideoSize(size.width, size.height),
                    ranges = map.getHighSpeedVideoFpsRangesFor(size).orEmpty().map {
                        FpsRange(it.lower, it.upper)
                    },
                )
            }.orEmpty()
            val normal = map.getOutputSizes(SurfaceTexture::class.java).orEmpty().map {
                VideoSize(it.width, it.height)
            }
            CameraOffer(
                id = id,
                facingBack = facing == CameraCharacteristics.LENS_FACING_BACK,
                highSpeed = highSpeed,
                normalSizes = normal,
            )
        }
    }

    private fun publish(phase: CameraPhase) {
        main.post { onPhase(phase) }
    }

    companion object {
        private const val TAG = "BallTracker"
    }
}

private data class RecorderAttempt(val containerFps: Int, val encoder: Int, val name: String)

private data class LensFacts(
    val focalLengthMm: Double? = null,
    val sensorWidthMm: Double? = null,
    val sensorHeightMm: Double? = null,
    val intrinsic: List<Double>? = null,
    val distortion: List<Double>? = null,
    val activeArray: List<Int>? = null,
    val preCorrectionArray: List<Int>? = null,
)

internal fun previewMatrix(
    viewWidth: Int,
    viewHeight: Int,
    bufferWidth: Int,
    bufferHeight: Int,
    rotation: Int,
): Matrix {
    val matrix = Matrix()
    val viewRect = RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
    val centerX = viewRect.centerX()
    val centerY = viewRect.centerY()
    if (rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270) {
        val bufferRect = RectF(0f, 0f, bufferHeight.toFloat(), bufferWidth.toFloat())
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)
        val scale = maxOf(
            viewHeight.toFloat() / bufferHeight,
            viewWidth.toFloat() / bufferWidth,
        )
        matrix.postScale(scale, scale, centerX, centerY)
        matrix.postRotate(90f * (rotation - 2), centerX, centerY)
    } else if (rotation == Surface.ROTATION_180) {
        matrix.postRotate(180f, centerX, centerY)
    } else {
        // Natural portrait. The high-speed buffer is landscape, so rotate it upright
        // with one scale on both axes. A non-uniform scale is what was squashing the picture.
        val bufferRect = RectF(0f, 0f, bufferHeight.toFloat(), bufferWidth.toFloat())
        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)
        val scale = maxOf(
            viewHeight.toFloat() / bufferWidth,
            viewWidth.toFloat() / bufferHeight,
        )
        matrix.postScale(scale, scale, centerX, centerY)
    }
    return matrix
}
