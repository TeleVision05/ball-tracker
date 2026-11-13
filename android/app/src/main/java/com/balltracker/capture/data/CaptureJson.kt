package com.balltracker.capture.data

import com.balltracker.capture.geometry.NormPoint
import com.balltracker.capture.geometry.StumpMarks
import java.util.Locale
import kotlin.math.abs

data class BallRecord(
    val id: String,
    val recordedAt: String,
    val videoFile: String,
    val setupFrameFile: String?,
    val encodedWidth: Int,
    val encodedHeight: Int,
    val captureFps: Int,
    val fpsRangeLower: Int,
    val fpsRangeUpper: Int,
    val fpsFixed: Boolean,
    val measuredFps: Double?,
    val containerFps: Int,
    val orientationHintDegrees: Int,
    val cameraId: String,
    val marks: StumpMarks,
    val wallClockSeconds: Double = 0.0,
    val frameCount: Int = 0,
    val durationSeconds: Double = 0.0,
    val maxGapSeconds: Double = 0.0,
    val longGapCount: Int = 0,
    val timestampsUs: List<Long> = emptyList(),
    val encoder: String? = null,
    val bitrate: Int? = null,
    val uprightWidth: Int? = null,
    val uprightHeight: Int? = null,
    val intrinsicSource: String? = null,
    val intrinsicMatrixBuffer: List<List<Double>>? = null,
    val intrinsicMatrixUpright: List<List<Double>>? = null,
    val focalLengthMm: Double? = null,
    val sensorWidthMm: Double? = null,
    val sensorHeightMm: Double? = null,
    val lensIntrinsicCalibration: List<Double>? = null,
    val lensDistortion: List<Double>? = null,
    val activeArray: List<Int>? = null,
    val preCorrectionActiveArray: List<Int>? = null,
    val cropRegion: List<Int>? = null,
    val cropChanged: Boolean = false,
    val focusDistanceDiopters: Double? = null,
    val exposureSampleCount: Int = 0,
    val exposureNsMin: Long? = null,
    val exposureNsMedian: Long? = null,
    val exposureNsMax: Long? = null,
    val isoMin: Int? = null,
    val isoMedian: Int? = null,
    val isoMax: Int? = null,
)

fun ballJson(record: BallRecord): String {
    val markers = linkedMapOf(
        "nearLeft" to listOf(record.marks.nearLeft.x, record.marks.nearLeft.y),
        "nearRight" to listOf(record.marks.nearRight.x, record.marks.nearRight.y),
        "farLeft" to listOf(record.marks.farLeft.x, record.marks.farLeft.y),
        "farRight" to listOf(record.marks.farRight.x, record.marks.farRight.y),
    )
    val body = linkedMapOf<String, Any?>(
        "version" to 2,
        "id" to record.id,
        "recordedAt" to record.recordedAt,
        "videoFile" to record.videoFile,
        "setupFrameFile" to record.setupFrameFile,
        "encodedWidth" to record.encodedWidth,
        "encodedHeight" to record.encodedHeight,
        "captureFps" to record.captureFps,
        "fpsRange" to listOf(record.fpsRangeLower, record.fpsRangeUpper),
        "fpsFixed" to record.fpsFixed,
        "measuredFps" to record.measuredFps,
        "containerFps" to record.containerFps,
        "orientationHintDegrees" to record.orientationHintDegrees,
        "cameraId" to record.cameraId,
        "encoder" to record.encoder,
        "bitrate" to record.bitrate,
        "wallClockSeconds" to record.wallClockSeconds,
        "frameCount" to record.frameCount,
        "durationSeconds" to record.durationSeconds,
        "maxGapSeconds" to record.maxGapSeconds,
        "longGapCount" to record.longGapCount,
        "uprightWidth" to record.uprightWidth,
        "uprightHeight" to record.uprightHeight,
        "timing" to "Use presentationTimestampsUs. Each value is that frame's container " +
            "presentation time in microseconds. When durationSeconds matches wallClockSeconds, " +
            "those times are real seconds and measuredFps is the real frame rate. Do not use " +
            "frameIndex/captureFps: the encoder can drop frames.",
        "presentationTimestampsUs" to record.timestampsUs,
        "camera" to linkedMapOf(
            "focalLengthMm" to record.focalLengthMm,
            "sensorSizeMm" to listOf(record.sensorWidthMm, record.sensorHeightMm),
            "focusDistanceDiopters" to record.focusDistanceDiopters,
            "lensIntrinsicCalibration" to record.lensIntrinsicCalibration,
            "lensDistortion" to record.lensDistortion,
            "lensDistortionNote" to "Raw Camera2 LENS_DISTORTION array. It is not an OpenCV distCoeffs vector.",
            "activeArray" to record.activeArray,
            "preCorrectionActiveArray" to record.preCorrectionActiveArray,
            "cropRegion" to record.cropRegion,
            "cropChangedDuringClip" to record.cropChanged,
            "intrinsicSource" to record.intrinsicSource,
            "intrinsicMatrixBuffer" to record.intrinsicMatrixBuffer,
            "intrinsicMatrixUpright" to record.intrinsicMatrixUpright,
            "intrinsicNote" to "intrinsicMatrixBuffer is for the encoded 1920x1080 buffer. " +
                "intrinsicMatrixUpright is for the picture after rotating that buffer clockwise " +
                "by orientationHintDegrees. Marker fractions use the upright picture.",
        ),
        "exposure" to linkedMapOf(
            "sampleCount" to record.exposureSampleCount,
            "exposureNsMin" to record.exposureNsMin,
            "exposureNsMedian" to record.exposureNsMedian,
            "exposureNsMax" to record.exposureNsMax,
            "isoMin" to record.isoMin,
            "isoMedian" to record.isoMedian,
            "isoMax" to record.isoMax,
        ),
        "markers" to markers,
        "markerMeaning" to "Outer stump bases. Near is the bowler's end, closer to the camera. " +
            "Fractions of the upright frame, origin top-left, x to the right, y down.",
    )
    return renderJson(body)
}

fun parseMarks(json: String): StumpMarks? {
    fun pair(key: String): NormPoint? {
        val match = Regex(
            """"$key"\s*:\s*\[\s*(-?[0-9.eE+]+)\s*,\s*(-?[0-9.eE+]+)\s*]""",
        ).find(json) ?: return null
        val x = match.groupValues[1].toDoubleOrNull() ?: return null
        val y = match.groupValues[2].toDoubleOrNull() ?: return null
        return NormPoint(x, y)
    }
    return StumpMarks(
        nearLeft = pair("nearLeft") ?: return null,
        nearRight = pair("nearRight") ?: return null,
        farLeft = pair("farLeft") ?: return null,
        farRight = pair("farRight") ?: return null,
    )
}

fun renderJson(value: Any?): String = when (value) {
    null -> "null"
    is String -> "\"" + escapeJson(value) + "\""
    is Boolean -> value.toString()
    is Int, is Long -> value.toString()
    is Double, is Float -> formatNumber(value.toDouble())
    is Number -> value.toString()
    is Map<*, *> -> value.entries.joinToString(prefix = "{", postfix = "}") { (key, item) ->
        "\"${escapeJson(key.toString())}\":${renderJson(item)}"
    }
    is List<*> -> value.joinToString(prefix = "[", postfix = "]") { renderJson(it) }
    else -> error("Cannot write $value as JSON")
}

private fun formatNumber(value: Double): String {
    if (value.isNaN() || value.isInfinite()) return "null"
    if (abs(value) < 1e15 && value % 1.0 == 0.0) return value.toLong().toString()
    return String.format(Locale.US, "%.5f", value).trimEnd('0').trimEnd('.')
}

private fun escapeJson(text: String): String = buildString {
    for (char in text) {
        when (char) {
            '\\' -> append("\\\\")
            '"' -> append("\\\"")
            '\n' -> append("\\n")
            else -> append(char)
        }
    }
}
