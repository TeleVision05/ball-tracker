package com.balltracker.capture.geometry

/**
 * Facts the tracker needs from a clip, kept as plain numbers so they can be tested
 * without a phone. Times are the container's presentation timestamps, which are the
 * real clock. Frame index divided by the requested frame rate is not, because the
 * encoder drops frames.
 */
data class ClipClock(
    val frameCount: Int,
    val durationSeconds: Double,
    val measuredFps: Double?,
    val maxGapSeconds: Double,
    val longGapCount: Int,
    val timestampsUs: List<Long>,
)

fun clipClock(timestampsUs: List<Long>, requestedFps: Int): ClipClock {
    val ordered = timestampsUs.filter { it >= 0L }.distinct().sorted()
    if (ordered.isEmpty()) {
        return ClipClock(0, 0.0, null, 0.0, 0, emptyList())
    }
    val spanUs = ordered.last() - ordered.first()
    val durationSeconds = spanUs / 1_000_000.0
    val measured = if (ordered.size > 1 && spanUs > 0L) {
        (ordered.size - 1).toDouble() / durationSeconds
    } else {
        null
    }
    val expectedUs = if (requestedFps > 0) 1_000_000.0 / requestedFps else 0.0
    var maxGapUs = 0L
    var longGaps = 0
    for (index in 1 until ordered.size) {
        val gap = ordered[index] - ordered[index - 1]
        if (gap > maxGapUs) maxGapUs = gap
        if (expectedUs > 0.0 && gap > expectedUs * 1.5) longGaps += 1
    }
    return ClipClock(
        frameCount = ordered.size,
        durationSeconds = durationSeconds,
        measuredFps = measured,
        maxGapSeconds = maxGapUs / 1_000_000.0,
        longGapCount = longGaps,
        timestampsUs = ordered,
    )
}

data class FittedCrop(val left: Int, val top: Int, val width: Int, val height: Int)

/**
 * The reported crop can be 4:3 while the video is 16:9. The camera then keeps the center
 * of that crop. Scale the lens from that centered rectangle, not from the whole sensor.
 */
fun fitCropToBuffer(
    cropLeft: Int,
    cropTop: Int,
    cropWidth: Int,
    cropHeight: Int,
    bufferWidth: Int,
    bufferHeight: Int,
): FittedCrop {
    if (cropWidth <= 0 || cropHeight <= 0 || bufferWidth <= 0 || bufferHeight <= 0) {
        return FittedCrop(cropLeft, cropTop, cropWidth, cropHeight)
    }
    val cropAspect = cropWidth.toDouble() / cropHeight
    val bufferAspect = bufferWidth.toDouble() / bufferHeight
    return if (cropAspect > bufferAspect * 1.01) {
        val usedWidth = (cropHeight * bufferAspect).toInt()
        FittedCrop(
            left = cropLeft + (cropWidth - usedWidth) / 2,
            top = cropTop,
            width = usedWidth,
            height = cropHeight,
        )
    } else if (cropAspect < bufferAspect * 0.99) {
        val usedHeight = (cropWidth / bufferAspect).toInt()
        FittedCrop(
            left = cropLeft,
            top = cropTop + (cropHeight - usedHeight) / 2,
            width = cropWidth,
            height = usedHeight,
        )
    } else {
        FittedCrop(cropLeft, cropTop, cropWidth, cropHeight)
    }
}

fun medianLong(values: List<Long>): Long? {
    if (values.isEmpty()) return null
    val sorted = values.sorted()
    return sorted[sorted.size / 2]
}

fun medianInt(values: List<Int>): Int? {
    if (values.isEmpty()) return null
    val sorted = values.sorted()
    return sorted[sorted.size / 2]
}

/** Focal length and principal point, in pixels of one image. */
data class PixelLens(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
)

/**
 * Map calibration pixels into the encoded buffer.
 *
 * [fx]..[cy] are in the pre-correction active array, which is what Camera2 reports
 * for LENS_INTRINSIC_CALIBRATION. The crop is in active-array pixels. The buffer is
 * that crop scaled to [bufferWidth] x [bufferHeight].
 */
fun scaleIntrinsicsToBuffer(
    fx: Double,
    fy: Double,
    cx: Double,
    cy: Double,
    preLeft: Int,
    preTop: Int,
    activeLeft: Int,
    activeTop: Int,
    cropLeft: Int,
    cropTop: Int,
    cropWidth: Int,
    cropHeight: Int,
    bufferWidth: Int,
    bufferHeight: Int,
): PixelLens? {
    if (cropWidth <= 0 || cropHeight <= 0 || bufferWidth <= 0 || bufferHeight <= 0) return null
    val cropX0 = (cropLeft - activeLeft) + preLeft
    val cropY0 = (cropTop - activeTop) + preTop
    val scaleX = bufferWidth.toDouble() / cropWidth
    val scaleY = bufferHeight.toDouble() / cropHeight
    return PixelLens(
        fx = fx * scaleX,
        fy = fy * scaleY,
        cx = (cx - cropX0) * scaleX,
        cy = (cy - cropY0) * scaleY,
    )
}

data class UprightLens(
    val lens: PixelLens,
    val width: Int,
    val height: Int,
)

/**
 * Turn a lens for the encoded buffer into a lens for the upright picture.
 * [clockwiseDegrees] is the same rotation the video file's orientation hint applies:
 * 90 clockwise turns a 1920x1080 buffer into a 1080x1920 picture.
 */
fun uprightLens(buffer: PixelLens, bufferWidth: Int, bufferHeight: Int, clockwiseDegrees: Int): UprightLens {
    val quarter = ((clockwiseDegrees % 360) + 360) % 360
    return when (quarter) {
        90 -> UprightLens(
            lens = PixelLens(
                fx = buffer.fy,
                fy = buffer.fx,
                cx = bufferHeight - buffer.cy,
                cy = buffer.cx,
            ),
            width = bufferHeight,
            height = bufferWidth,
        )
        180 -> UprightLens(
            lens = PixelLens(
                fx = buffer.fx,
                fy = buffer.fy,
                cx = bufferWidth - buffer.cx,
                cy = bufferHeight - buffer.cy,
            ),
            width = bufferWidth,
            height = bufferHeight,
        )
        270 -> UprightLens(
            lens = PixelLens(
                fx = buffer.fy,
                fy = buffer.fx,
                cx = buffer.cy,
                cy = bufferWidth - buffer.cx,
            ),
            width = bufferHeight,
            height = bufferWidth,
        )
        else -> UprightLens(buffer, bufferWidth, bufferHeight)
    }
}

fun lensMatrix(lens: PixelLens): List<List<Double>> = listOf(
    listOf(lens.fx, 0.0, lens.cx),
    listOf(0.0, lens.fy, lens.cy),
    listOf(0.0, 0.0, 1.0),
)

data class DescribedCamera(
    val buffer: PixelLens?,
    val upright: PixelLens?,
    val uprightWidth: Int,
    val uprightHeight: Int,
    val source: String?,
)

/**
 * Prefer the device's intrinsic calibration. If the phone does not report one,
 * estimate the same pixels from the focal length and the sensor size.
 * Rectangles are left, top, right, bottom.
 */
fun describeCamera(
    intrinsic: List<Double>?,
    focalLengthMm: Double?,
    sensorWidthMm: Double?,
    sensorHeightMm: Double?,
    active: List<Int>?,
    pre: List<Int>?,
    crop: List<Int>?,
    bufferWidth: Int,
    bufferHeight: Int,
    clockwiseDegrees: Int,
): DescribedCamera {
    val activeLeft = active?.getOrNull(0) ?: 0
    val activeTop = active?.getOrNull(1) ?: 0
    val activeRight = active?.getOrNull(2)
    val activeBottom = active?.getOrNull(3)
    val activeWidth = if (activeRight != null) activeRight - activeLeft else 0
    val activeHeight = if (activeBottom != null) activeBottom - activeTop else 0
    val cropLeft: Int
    val cropTop: Int
    val cropWidth: Int
    val cropHeight: Int
    if (crop != null && crop.size >= 4 && crop[2] > crop[0] && crop[3] > crop[1]) {
        cropLeft = crop[0]
        cropTop = crop[1]
        cropWidth = crop[2] - crop[0]
        cropHeight = crop[3] - crop[1]
    } else if (activeWidth > 0 && activeHeight > 0) {
        cropLeft = activeLeft
        cropTop = activeTop
        cropWidth = activeWidth
        cropHeight = activeHeight
    } else {
        cropLeft = 0
        cropTop = 0
        cropWidth = bufferWidth
        cropHeight = bufferHeight
    }
    val fitted = fitCropToBuffer(cropLeft, cropTop, cropWidth, cropHeight, bufferWidth, bufferHeight)
    val preLeft = pre?.getOrNull(0) ?: 0
    val preTop = pre?.getOrNull(1) ?: 0
    val fromDevice = if (intrinsic != null && intrinsic.size >= 4) {
        scaleIntrinsicsToBuffer(
            fx = intrinsic[0],
            fy = intrinsic[1],
            cx = intrinsic[2],
            cy = intrinsic[3],
            preLeft = preLeft,
            preTop = preTop,
            activeLeft = activeLeft,
            activeTop = activeTop,
            cropLeft = fitted.left,
            cropTop = fitted.top,
            cropWidth = fitted.width,
            cropHeight = fitted.height,
            bufferWidth = bufferWidth,
            bufferHeight = bufferHeight,
        )
    } else {
        null
    }
    val fromFocal = if (
        fromDevice == null &&
        focalLengthMm != null &&
        sensorWidthMm != null &&
        sensorHeightMm != null &&
        sensorWidthMm > 0.0 &&
        sensorHeightMm > 0.0 &&
        activeWidth > 0 &&
        activeHeight > 0
    ) {
        scaleIntrinsicsToBuffer(
            fx = focalLengthMm / sensorWidthMm * activeWidth,
            fy = focalLengthMm / sensorHeightMm * activeHeight,
            cx = activeWidth / 2.0,
            cy = activeHeight / 2.0,
            preLeft = 0,
            preTop = 0,
            activeLeft = 0,
            activeTop = 0,
            cropLeft = fitted.left - activeLeft,
            cropTop = fitted.top - activeTop,
            cropWidth = fitted.width,
            cropHeight = fitted.height,
            bufferWidth = bufferWidth,
            bufferHeight = bufferHeight,
        )
    } else {
        null
    }
    val bufferLens = fromDevice ?: fromFocal
    val sizeOnly = uprightLens(
        PixelLens(1.0, 1.0, bufferWidth / 2.0, bufferHeight / 2.0),
        bufferWidth,
        bufferHeight,
        clockwiseDegrees,
    )
    if (bufferLens == null) {
        return DescribedCamera(null, null, sizeOnly.width, sizeOnly.height, null)
    }
    val upright = uprightLens(bufferLens, bufferWidth, bufferHeight, clockwiseDegrees)
    val source = if (fromDevice != null) "lensIntrinsicCalibration" else "focalLength"
    return DescribedCamera(bufferLens, upright.lens, upright.width, upright.height, source)
}
