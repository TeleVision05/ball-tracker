package com.balltracker.capture

import com.balltracker.capture.camera.CameraOffer
import com.balltracker.capture.camera.FpsRange
import com.balltracker.capture.camera.HighSpeedOption
import com.balltracker.capture.camera.VideoSize
import com.balltracker.capture.camera.chooseProfile
import com.balltracker.capture.data.BallRecord
import com.balltracker.capture.data.ballFacts
import com.balltracker.capture.data.ballJson
import com.balltracker.capture.data.clipLimitSeconds
import com.balltracker.capture.data.parseMarks
import com.balltracker.capture.geometry.PixelLens
import com.balltracker.capture.geometry.clampMarker
import com.balltracker.capture.geometry.clipClock
import com.balltracker.capture.geometry.defaultStumpMarks
import com.balltracker.capture.geometry.describeCamera
import com.balltracker.capture.geometry.fitFrame
import com.balltracker.capture.geometry.orientationHintDegrees
import com.balltracker.capture.geometry.uprightAspect
import com.balltracker.capture.geometry.uprightLens
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureLogicTest {
    @Test
    fun s23UltraPrefersFixed1080p240() {
        val phone = CameraOffer(
            id = "0",
            facingBack = true,
            highSpeed = listOf(
                HighSpeedOption(
                    VideoSize(1920, 1080),
                    listOf(FpsRange(120, 120), FpsRange(240, 240), FpsRange(30, 240)),
                ),
                HighSpeedOption(VideoSize(1280, 720), listOf(FpsRange(240, 240))),
            ),
            normalSizes = listOf(VideoSize(1920, 1080)),
        )
        val chosen = chooseProfile(listOf(phone))
        assertEquals(1920, chosen?.width)
        assertEquals(1080, chosen?.height)
        assertEquals(240, chosen?.captureFps)
        assertEquals(true, chosen?.fixedFps)
        assertEquals(true, chosen?.slowMotion)
        assertNull(chosen?.statusLine())
    }

    @Test
    fun fallsBackWhen240IsMissing() {
        val phone = CameraOffer(
            id = "0",
            facingBack = true,
            highSpeed = listOf(
                HighSpeedOption(VideoSize(1920, 1080), listOf(FpsRange(120, 120))),
            ),
            normalSizes = listOf(VideoSize(1920, 1080)),
        )
        val chosen = chooseProfile(listOf(phone))
        assertEquals(120, chosen?.captureFps)
        assertEquals("120 was the fastest this phone offered", chosen?.statusLine())
    }

    @Test
    fun normalFrameRateWhenThereIsNoSlowMotion() {
        val phone = CameraOffer(
            id = "0",
            facingBack = true,
            highSpeed = emptyList(),
            normalSizes = listOf(VideoSize(1280, 720), VideoSize(1920, 1080)),
        )
        val chosen = chooseProfile(listOf(phone))
        assertEquals(1920, chosen?.width)
        assertEquals(30, chosen?.captureFps)
        assertEquals(false, chosen?.slowMotion)
    }

    @Test
    fun portraitFrameIsTheBufferTurnedOnItsSide() {
        assertEquals(16f / 9f, uprightAspect(1920, 1080, displayPortrait = false), 0.001f)
        assertEquals(9f / 16f, uprightAspect(1920, 1080, displayPortrait = true), 0.001f)
    }

    @Test
    fun landscapeHintOnAPortraitSensorIsZero() {
        assertEquals(0, orientationHintDegrees(sensorOrientation = 90, displayDegrees = 90))
        assertEquals(90, orientationHintDegrees(sensorOrientation = 90, displayDegrees = 0))
    }

    @Test
    fun frameFitsInsideTheViewWithoutCropping() {
        val rect = fitFrame(viewWidth = 2000f, viewHeight = 900f, frameWidth = 1920f, frameHeight = 1080f)
        assertEquals(900f / 1080f * 1920f, rect.width, 0.1f)
        assertEquals(900f, rect.height, 0.1f)
        assertTrue(rect.left > 0f)
        assertEquals(0f, rect.top, 0.1f)
    }

    @Test
    fun markersStayInsideTheFrame() {
        val clamped = clampMarker(-1.0, 2.0)
        assertEquals(0.02, clamped.x, 0.0001)
        assertEquals(0.98, clamped.y, 0.0001)
    }

    @Test
    fun sidecarRoundTripsTheFourBases() {
        val marks = defaultStumpMarks()
        val json = ballJson(
            BallRecord(
                id = "ball-0001",
                recordedAt = "2026-10-06T16:00:00-07:00",
                videoFile = "ball-0001.mp4",
                setupFrameFile = "setup-frame.jpg",
                encodedWidth = 1920,
                encodedHeight = 1080,
                captureFps = 240,
                fpsRangeLower = 240,
                fpsRangeUpper = 240,
                fpsFixed = true,
                measuredFps = 239.5,
                containerFps = 240,
                orientationHintDegrees = 0,
                cameraId = "0",
                marks = marks,
            ),
        )
        assertEquals(marks, parseMarks(json))
        assertTrue(json.contains("\"captureFps\":240"))
        assertTrue(json.contains("presentationTimestampsUs"))
        assertTrue(!json.contains("i/captureFps"))
    }

    @Test
    fun droppedFramesDoNotChangeTheRealClock() {
        val clock = clipClock(listOf(0L, 4_167L, 8_334L, 29_167L), requestedFps = 240)
        assertEquals(4, clock.frameCount)
        assertEquals(1, clock.longGapCount)
        assertEquals(0.020833, clock.maxGapSeconds, 0.00001)
        assertEquals(102.8, clock.measuredFps ?: 0.0, 0.2)
    }

    @Test
    fun portraitRotationPutsThePrincipalPointOnTheUprightFrame() {
        val upright = uprightLens(
            PixelLens(fx = 1000.0, fy = 1100.0, cx = 960.0, cy = 540.0),
            bufferWidth = 1920,
            bufferHeight = 1080,
            clockwiseDegrees = 90,
        )
        assertEquals(1080, upright.width)
        assertEquals(1920, upright.height)
        assertEquals(1100.0, upright.lens.fx, 0.01)
        assertEquals(1000.0, upright.lens.fy, 0.01)
        assertEquals(540.0, upright.lens.cx, 0.01)
        assertEquals(960.0, upright.lens.cy, 0.01)
    }

    @Test
    fun intrinsicsScaleFromTheSensorCropIntoTheBuffer() {
        val described = describeCamera(
            intrinsic = listOf(4000.0, 4000.0, 2000.0, 1500.0, 0.0),
            focalLengthMm = null,
            sensorWidthMm = null,
            sensorHeightMm = null,
            active = listOf(0, 0, 4000, 3000),
            pre = listOf(0, 0, 4000, 3000),
            crop = listOf(0, 0, 4000, 3000),
            bufferWidth = 2000,
            bufferHeight = 1500,
            clockwiseDegrees = 0,
        )
        assertEquals("lensIntrinsicCalibration", described.source)
        assertEquals(2000.0, described.buffer?.fx ?: 0.0, 0.01)
        assertEquals(1000.0, described.upright?.cx ?: 0.0, 0.01)
        assertEquals(750.0, described.upright?.cy ?: 0.0, 0.01)
    }

    @Test
    fun sidecarSummaryKeepsTheMeasuredRateAndShutter() {
        val json = """
            {"measuredFps":225.4,"captureFps":240,"durationSeconds":1.33,
            "frameCount":300,"maxGapSeconds":0.0208,"encoder":"hevc",
            "focalLengthMm":6.3,"exposureNsMedian":4001007,"isoMedian":640,
            "encodedWidth":1920,"encodedHeight":1080,"orientationHintDegrees":90,
            "recordedAt":"2026-10-08T16:00:00-07:00"}
        """.trimIndent()
        val facts = ballFacts(json)
        assertEquals(225.4, facts.measuredFps ?: 0.0, 0.01)
        assertEquals(300, facts.frameCount)
        assertEquals(4001007L, facts.exposureNsMedian)
        assertEquals("hevc", facts.encoder)
        assertEquals(45, clipLimitSeconds(90))
        assertEquals(15, clipLimitSeconds(15))
    }

    @Test
    fun wideVideoCropsTheTallSensorFromTheCenter() {
        val described = describeCamera(
            intrinsic = listOf(4000.0, 4000.0, 2040.0, 1530.0, 0.0),
            focalLengthMm = null,
            sensorWidthMm = null,
            sensorHeightMm = null,
            active = listOf(0, 0, 4080, 3060),
            pre = listOf(0, 0, 4080, 3060),
            crop = listOf(0, 0, 4080, 3060),
            bufferWidth = 1920,
            bufferHeight = 1080,
            clockwiseDegrees = 0,
        )
        assertEquals(1882.35, described.buffer?.fx ?: 0.0, 0.2)
        assertEquals(1882.35, described.buffer?.fy ?: 0.0, 0.2)
        assertEquals(960.0, described.buffer?.cx ?: 0.0, 1.0)
        assertEquals(540.0, described.buffer?.cy ?: 0.0, 1.0)
    }
}
