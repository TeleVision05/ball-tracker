package com.balltracker.capture.data

/** The fields worth showing for one saved ball. Times and lens numbers come from its JSON. */
data class BallFacts(
    val measuredFps: Double? = null,
    val captureFps: Int? = null,
    val durationSeconds: Double? = null,
    val frameCount: Int? = null,
    val maxGapSeconds: Double? = null,
    val encoder: String? = null,
    val focalLengthMm: Double? = null,
    val exposureNsMedian: Long? = null,
    val isoMedian: Int? = null,
    val encodedWidth: Int? = null,
    val encodedHeight: Int? = null,
    val orientationHintDegrees: Int? = null,
    val recordedAt: String? = null,
)

fun ballFacts(json: String): BallFacts = BallFacts(
    measuredFps = jsonNumber(json, "measuredFps"),
    captureFps = jsonNumber(json, "captureFps")?.toInt(),
    durationSeconds = jsonNumber(json, "durationSeconds"),
    frameCount = jsonNumber(json, "frameCount")?.toInt(),
    maxGapSeconds = jsonNumber(json, "maxGapSeconds"),
    encoder = jsonString(json, "encoder"),
    focalLengthMm = jsonNumber(json, "focalLengthMm"),
    exposureNsMedian = jsonNumber(json, "exposureNsMedian")?.toLong(),
    isoMedian = jsonNumber(json, "isoMedian")?.toInt(),
    encodedWidth = jsonNumber(json, "encodedWidth")?.toInt(),
    encodedHeight = jsonNumber(json, "encodedHeight")?.toInt(),
    orientationHintDegrees = jsonNumber(json, "orientationHintDegrees")?.toInt(),
    recordedAt = jsonString(json, "recordedAt"),
)

fun clipLimitSeconds(value: Int): Int = when (value) {
    15, 30 -> value
    else -> 45
}

private fun jsonNumber(json: String, key: String): Double? {
    val match = Regex(""""$key"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)""").find(json) ?: return null
    return match.groupValues[1].toDoubleOrNull()
}

private fun jsonString(json: String, key: String): String? {
    val match = Regex(""""$key"\s*:\s*"([^"]*)"""").find(json) ?: return null
    return match.groupValues[1]
}
