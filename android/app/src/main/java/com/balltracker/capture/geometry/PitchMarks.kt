package com.balltracker.capture.geometry

/**
 * Four outer stump bases, in the upright frame the person filmed.
 * Near is the bowler's end, the set closer to the phone.
 * Coordinates are fractions of that frame: origin top-left, x right, y down.
 */
data class NormPoint(val x: Double, val y: Double)

data class StumpMarks(
    val nearLeft: NormPoint,
    val nearRight: NormPoint,
    val farLeft: NormPoint,
    val farRight: NormPoint,
)

fun defaultStumpMarks(): StumpMarks = StumpMarks(
    nearLeft = NormPoint(0.33, 0.78),
    nearRight = NormPoint(0.67, 0.78),
    farLeft = NormPoint(0.43, 0.38),
    farRight = NormPoint(0.57, 0.38),
)

fun clampMarker(x: Double, y: Double): NormPoint = NormPoint(
    x.coerceIn(0.02, 0.98),
    y.coerceIn(0.02, 0.98),
)

/** Letterboxed rectangle of a camera frame inside a view. */
data class FrameRect(
    val left: Float,
    val top: Float,
    val width: Float,
    val height: Float,
)

fun fitFrame(
    viewWidth: Float,
    viewHeight: Float,
    frameWidth: Float,
    frameHeight: Float,
): FrameRect {
    if (viewWidth <= 0f || viewHeight <= 0f || frameWidth <= 0f || frameHeight <= 0f) {
        return FrameRect(0f, 0f, 0f, 0f)
    }
    val scale = minOf(viewWidth / frameWidth, viewHeight / frameHeight)
    val width = frameWidth * scale
    val height = frameHeight * scale
    return FrameRect(
        left = (viewWidth - width) / 2f,
        top = (viewHeight - height) / 2f,
        width = width,
        height = height,
    )
}

/**
 * Degrees a player should rotate the encoded buffer so it matches the screen.
 * [displayDegrees] is 0, 90, 180, or 270 from the default display rotation.
 */
fun orientationHintDegrees(sensorOrientation: Int, displayDegrees: Int): Int {
    return (sensorOrientation - displayDegrees + 360) % 360
}

/** Width / height of the frame as it should appear on screen. The buffer itself stays landscape. */
fun uprightAspect(bufferWidth: Int, bufferHeight: Int, displayPortrait: Boolean): Float {
    val wide = maxOf(bufferWidth, bufferHeight).toFloat()
    val tall = minOf(bufferWidth, bufferHeight).toFloat()
    if (wide <= 0f || tall <= 0f) return if (displayPortrait) 9f / 16f else 16f / 9f
    val landscape = wide / tall
    return if (displayPortrait) 1f / landscape else landscape
}

fun displayRotationDegrees(surfaceRotation: Int): Int = when (surfaceRotation) {
    1 -> 90
    2 -> 180
    3 -> 270
    else -> 0
}
