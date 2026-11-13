package com.balltracker.capture.ui

import android.content.res.Configuration
import android.view.HapticFeedbackConstants
import android.view.TextureView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.balltracker.capture.camera.CameraPhase
import com.balltracker.capture.camera.ChosenProfile
import com.balltracker.capture.geometry.FrameRect
import com.balltracker.capture.geometry.NormPoint
import com.balltracker.capture.geometry.StumpMarks
import com.balltracker.capture.geometry.uprightAspect
import kotlin.math.roundToInt

@Composable
fun CaptureScreen(
    state: CaptureUiState,
    onAttach: (TextureView) -> Unit,
    onDetach: (TextureView) -> Unit,
    onNudge: (MarkerId, Double, Double) -> Unit,
    onMarksSettled: () -> Unit,
    onRecord: () -> Unit,
    onAllowCamera: () -> Unit,
    onOrientationChanged: () -> Unit,
    onOpen: (Place) -> Unit,
    onDelete: (String) -> Unit,
    onClipSeconds: (Int) -> Unit,
    onKeepPreview: (Boolean) -> Unit,
) {
    val orientation = LocalConfiguration.current.orientation
    LaunchedEffect(orientation) { onOrientationChanged() }
    val recording = state.phase is CameraPhase.Recording
    Column(Modifier.fillMaxSize().background(Pitch.ground)) {
        Box(Modifier.weight(1f)) {
            when (val place = state.place) {
                Place.Film -> Viewfinder(
                    state = state,
                    onAttach = onAttach,
                    onDetach = onDetach,
                    onNudge = onNudge,
                    onMarksSettled = onMarksSettled,
                    onRecord = onRecord,
                    onAllowCamera = onAllowCamera,
                )
                Place.Balls -> BallList(state.balls) { onOpen(Place.Ball(it)) }
                is Place.Ball -> BallDetail(
                    ball = state.balls.find { it.id == place.id },
                    onBack = { onOpen(Place.Balls) },
                    onDelete = onDelete,
                )
                Place.Settings -> SettingsScreen(
                    options = state.options,
                    onClipSeconds = onClipSeconds,
                    onKeepPreview = onKeepPreview,
                )
            }
        }
        if (!recording && state.place !is Place.Ball) {
            SectionBar(current = state.place, onOpen = onOpen)
        }
    }
}

@Composable
private fun Viewfinder(
    state: CaptureUiState,
    onAttach: (TextureView) -> Unit,
    onDetach: (TextureView) -> Unit,
    onNudge: (MarkerId, Double, Double) -> Unit,
    onMarksSettled: () -> Unit,
    onRecord: () -> Unit,
    onAllowCamera: () -> Unit,
) {
    val view = LocalView.current
    val recording = state.phase is CameraPhase.Recording
    Box(Modifier.fillMaxSize().background(Pitch.ground)) {
        if (state.permissionGranted) {
            LivePreview(
                state = state,
                onAttach = onAttach,
                onDetach = onDetach,
                onNudge = onNudge,
                onMarksSettled = onMarksSettled,
            )
        }
        Column(
            Modifier
                .fillMaxSize()
                .windowInsetsPadding(
                    if (recording) {
                        androidx.compose.foundation.layout.WindowInsets.safeDrawing
                    } else {
                        androidx.compose.foundation.layout.WindowInsets.statusBars
                    },
                )
                .padding(horizontal = 28.dp, vertical = 16.dp),
        ) {
            if (!recording) {
                TopBar(state)
            } else {
                Text(
                    text = formatElapsed(state.elapsedMs),
                    color = Pitch.stump,
                    fontSize = 28.sp,
                    style = labelShadow(),
                )
                Text(
                    text = "The picture holds still so every frame is kept.",
                    color = Pitch.stump,
                    fontSize = 14.sp,
                    style = labelShadow(),
                )
            }
            Spacer(Modifier.weight(1f))
            when {
                !state.permissionGranted -> PermissionLine(onAllowCamera)
                state.phase is CameraPhase.Failed -> Text(
                    text = state.phase.message,
                    color = Pitch.stump,
                    style = labelShadow(),
                )
                recording -> RecordRow(
                    recording = true,
                    canRecord = false,
                    onRecord = {
                        view.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
                        onRecord()
                    },
                )
                else -> RecordRow(
                    recording = false,
                    canRecord = state.phase is CameraPhase.Previewing && !state.busy && state.marks != null,
                    onRecord = {
                        view.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
                        onRecord()
                    },
                )
            }
        }
    }
}

@Composable
private fun TopBar(state: CaptureUiState) {
    val profile = when (val phase = state.phase) {
        is CameraPhase.Previewing -> phase.profile
        is CameraPhase.Recording -> phase.profile
        else -> null
    }
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            if (profile != null) {
                Text(
                    text = profile.captureFps.toString(),
                    color = Pitch.stump,
                    fontSize = 28.sp,
                    fontFamily = FontFamily.Serif,
                    style = labelShadow(),
                )
                val note = (state.phase as? CameraPhase.Previewing)?.note
                val status = note ?: profile.statusLine()
                if (status != null) {
                    Text(text = status, color = Pitch.stump, fontSize = 14.sp, style = labelShadow())
                }
            }
        }
    }
}

@Composable
private fun SectionBar(current: Place, onOpen: (Place) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(Pitch.ground)
            .windowInsetsPadding(androidx.compose.foundation.layout.WindowInsets.navigationBars)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
    ) {
        SectionTab("Film", current is Place.Film, "Open filming") { onOpen(Place.Film) }
        SectionTab("Balls", current is Place.Balls, "Open balls") { onOpen(Place.Balls) }
        SectionTab("Settings", current is Place.Settings, "Open settings") { onOpen(Place.Settings) }
    }
}

@Composable
private fun SectionTab(label: String, selected: Boolean, description: String, onClick: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clickable(onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 6.dp)
            .semantics { contentDescription = description },
    ) {
        Text(
            text = label,
            color = if (selected) Pitch.stump else Pitch.stump.copy(alpha = 0.45f),
            fontSize = 16.sp,
        )
        Box(
            Modifier
                .padding(top = 4.dp)
                .size(width = 18.dp, height = 2.dp)
                .background(if (selected) Pitch.cherry else Pitch.cherry.copy(alpha = 0f)),
        )
    }
}

@Composable
private fun LivePreview(
    state: CaptureUiState,
    onAttach: (TextureView) -> Unit,
    onDetach: (TextureView) -> Unit,
    onNudge: (MarkerId, Double, Double) -> Unit,
    onMarksSettled: () -> Unit,
) {
    val portrait = LocalConfiguration.current.orientation == Configuration.ORIENTATION_PORTRAIT
    val profile = when (val phase = state.phase) {
        is CameraPhase.Previewing -> phase.profile
        is CameraPhase.Recording -> phase.profile
        else -> null
    }
    val aspect = uprightAspect(
        bufferWidth = profile?.width ?: 1920,
        bufferHeight = profile?.height ?: 1080,
        displayPortrait = portrait,
    )
    BoxWithConstraints(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        val frameRatio = aspect
        val viewRatio = if (maxHeight.value == 0f) frameRatio else maxWidth.value / maxHeight.value
        val width = if (frameRatio > viewRatio) maxWidth else maxHeight * frameRatio
        val height = if (frameRatio > viewRatio) maxWidth / frameRatio else maxHeight
        Box(Modifier.size(width, height)) {
            AndroidView(
                factory = { context -> TextureView(context).also(onAttach) },
                onRelease = onDetach,
                modifier = Modifier.fillMaxSize(),
            )
            MarkOverlay(state.marks ?: state.draft, onNudge, onMarksSettled)
        }
    }
}

@Composable
private fun MarkOverlay(
    marks: StumpMarks,
    onNudge: (MarkerId, Double, Double) -> Unit,
    onMarksSettled: () -> Unit,
) {
    var size by remember { mutableStateOf(IntSize.Zero) }
    val rect = FrameRect(0f, 0f, size.width.toFloat(), size.height.toFloat())
    Box(Modifier.fillMaxSize().onSizeChanged { size = it }) {
        PitchOutline(rect, marks)
        MarkerHandle(MarkerId.NearLeft, "Near left", Pitch.stump, true, marks.nearLeft, rect, onNudge, onMarksSettled)
        MarkerHandle(MarkerId.NearRight, "Near right", Pitch.stump, true, marks.nearRight, rect, onNudge, onMarksSettled)
        MarkerHandle(MarkerId.FarLeft, "Far left", Pitch.far, false, marks.farLeft, rect, onNudge, onMarksSettled)
        MarkerHandle(MarkerId.FarRight, "Far right", Pitch.far, false, marks.farRight, rect, onNudge, onMarksSettled)
    }
}

@Composable
private fun RecordRow(
    recording: Boolean,
    canRecord: Boolean,
    onRecord: () -> Unit,
) {
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        RecordButton(recording = recording, enabled = recording || canRecord, onClick = onRecord)
    }
}

@Composable
private fun RecordButton(recording: Boolean, enabled: Boolean, onClick: () -> Unit) {
    Box(
        Modifier
            .size(84.dp)
            .clip(CircleShape)
            .background(if (enabled) Pitch.cherry else Pitch.cherry.copy(alpha = 0.4f))
            .clickable(enabled = enabled, onClick = onClick)
            .semantics { contentDescription = if (recording) "Stop" else "Record" },
        contentAlignment = Alignment.Center,
    ) {
        if (recording) {
            Box(
                Modifier
                    .size(26.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(Pitch.stump),
            )
        }
    }
}

@Composable
private fun PermissionLine(onAllow: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = "Camera permission is required to film.",
            color = Pitch.stump,
            modifier = Modifier.weight(1f),
        )
        ActionButton(label = "Allow the camera", enabled = true, onClick = onAllow)
    }
}

@Composable
private fun PitchOutline(rect: FrameRect, marks: StumpMarks) {
    if (rect.width <= 0f) return
    Canvas(Modifier.fillMaxSize()) {
        fun point(marker: NormPoint) = Offset(
            rect.left + (marker.x * rect.width).toFloat(),
            rect.top + (marker.y * rect.height).toFloat(),
        )
        val path = Path().apply {
            val nearLeft = point(marks.nearLeft)
            moveTo(nearLeft.x, nearLeft.y)
            listOf(marks.nearRight, marks.farRight, marks.farLeft, marks.nearLeft).forEach {
                val next = point(it)
                lineTo(next.x, next.y)
            }
        }
        drawPath(path, color = Color.White.copy(alpha = 0.55f), style = Stroke(width = 2.dp.toPx()))
    }
}

@Composable
private fun MarkerHandle(
    id: MarkerId,
    label: String,
    color: Color,
    labelAbove: Boolean,
    point: NormPoint,
    rect: FrameRect,
    onNudge: (MarkerId, Double, Double) -> Unit,
    onMarksSettled: () -> Unit,
) {
    if (rect.width <= 0f) return
    val density = LocalDensity.current
    val touch = 56.dp
    val touchPx = with(density) { touch.toPx() }
    val cx = rect.left + (point.x * rect.width).toFloat()
    val cy = rect.top + (point.y * rect.height).toFloat()
    Box(
        Modifier
            .offset { IntOffset((cx - touchPx / 2f).roundToInt(), (cy - touchPx / 2f).roundToInt()) }
            .size(touch)
            .pointerInput(id, rect.width, rect.height) {
                detectDragGestures(
                    onDragEnd = { onMarksSettled() },
                    onDragCancel = { onMarksSettled() },
                ) { change, drag ->
                    change.consume()
                    onNudge(id, drag.x / rect.width.toDouble(), drag.y / rect.height.toDouble())
                }
            }
            .semantics { contentDescription = "$label stump" },
    ) {
        MarkerDot(color, Modifier.align(Alignment.Center))
        Text(
            text = label,
            color = color,
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            style = labelShadow(),
            modifier = Modifier
                .align(if (labelAbove) Alignment.TopCenter else Alignment.BottomCenter)
                .offset(y = if (labelAbove) (-16).dp else 16.dp),
        )
    }
}

@Composable
private fun MarkerDot(color: Color, modifier: Modifier = Modifier) {
    Canvas(modifier.size(22.dp)) {
        drawCircle(color, radius = size.minDimension / 2f, style = Stroke(width = 3.dp.toPx()))
        drawCircle(color, radius = 2.dp.toPx())
    }
}

@Composable
private fun ActionButton(label: String, enabled: Boolean, onClick: () -> Unit) {
    Text(
        text = label,
        color = if (enabled) Pitch.ink else Pitch.ink.copy(alpha = 0.4f),
        modifier = Modifier
            .clip(RoundedCornerShape(28.dp))
            .background(if (enabled) Pitch.stump else Pitch.stump.copy(alpha = 0.4f))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 22.dp, vertical = 14.dp),
    )
}

private fun labelShadow() = TextStyle(
    shadow = Shadow(color = Color.Black.copy(alpha = 0.7f), offset = Offset(0f, 1f), blurRadius = 6f),
)

private fun formatElapsed(elapsedMs: Long): String {
    val seconds = elapsedMs / 1000.0
    return String.format(java.util.Locale.US, "%.1fs", seconds)
}

