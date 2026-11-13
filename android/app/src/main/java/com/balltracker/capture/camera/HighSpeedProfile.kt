package com.balltracker.capture.camera

data class VideoSize(val width: Int, val height: Int)

data class FpsRange(val lower: Int, val upper: Int) {
    val fixed: Boolean get() = lower == upper
}

data class HighSpeedOption(val size: VideoSize, val ranges: List<FpsRange>)

data class CameraOffer(
    val id: String,
    val facingBack: Boolean,
    val highSpeed: List<HighSpeedOption>,
    val normalSizes: List<VideoSize>,
)

data class ChosenProfile(
    val cameraId: String,
    val width: Int,
    val height: Int,
    val fpsLower: Int,
    val fpsUpper: Int,
    val slowMotion: Boolean,
) {
    val captureFps: Int get() = fpsUpper
    val fixedFps: Boolean get() = fpsLower == fpsUpper

    fun statusLine(): String? = when {
        slowMotion && captureFps == 240 && fixedFps -> null
        slowMotion && captureFps != 240 -> "$captureFps was the fastest this phone offered"
        slowMotion && !fixedFps -> "The phone may not hold $captureFps the whole time"
        else -> "Slow motion is not available. This clip is $captureFps frames a second"
    }
}

private data class Wanted(val width: Int, val height: Int, val fps: Int)

private val slowMotionPreference = listOf(
    Wanted(1920, 1080, 240),
    Wanted(1280, 720, 240),
    Wanted(1920, 1080, 120),
    Wanted(1280, 720, 120),
)

/**
 * Prefer a fixed 240 fps 1080p range on the back camera. The S23 Ultra advertises that.
 * A fixed range is one whose lower and upper bounds are equal, so the shutter cannot
 * drift down to 30 fps inside a wide range.
 */
fun chooseProfile(cameras: List<CameraOffer>): ChosenProfile? {
    val backs = cameras.filter { it.facingBack }.ifEmpty { cameras }
    for (wanted in slowMotionPreference) {
        val chosen = backs.firstNotNullOfOrNull { camera ->
            matchHighSpeed(camera, wanted)
        }
        if (chosen != null) return chosen
    }
    val normalOrder = listOf(VideoSize(1920, 1080), VideoSize(1280, 720))
    for (camera in backs) {
        val size = normalOrder.firstOrNull { wanted ->
            camera.normalSizes.any { it.width == wanted.width && it.height == wanted.height }
        } ?: camera.normalSizes.maxByOrNull { it.width.toLong() * it.height }
        if (size != null) {
            return ChosenProfile(
                cameraId = camera.id,
                width = size.width,
                height = size.height,
                fpsLower = 30,
                fpsUpper = 30,
                slowMotion = false,
            )
        }
    }
    return null
}

private fun matchHighSpeed(camera: CameraOffer, wanted: Wanted): ChosenProfile? {
    val option = camera.highSpeed.firstOrNull {
        it.size.width == wanted.width && it.size.height == wanted.height
    } ?: return null
    val fixed = option.ranges.firstOrNull { it.fixed && it.upper == wanted.fps }
    val flexible = option.ranges.firstOrNull { it.upper == wanted.fps && it.lower <= wanted.fps }
    val range = fixed ?: flexible ?: return null
    return ChosenProfile(
        cameraId = camera.id,
        width = wanted.width,
        height = wanted.height,
        fpsLower = range.lower,
        fpsUpper = range.upper,
        slowMotion = true,
    )
}
