package com.balltracker.capture.ui

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.widget.VideoView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import com.balltracker.capture.data.BallFacts
import com.balltracker.capture.data.SavedBall
import com.balltracker.capture.data.ballFacts
import java.util.Locale

@Composable
fun BallList(
    balls: List<SavedBall>,
    onOpen: (String) -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .windowInsetsPadding(androidx.compose.foundation.layout.WindowInsets.statusBars)
            .padding(horizontal = 24.dp),
    ) {
        Text(
            text = "Balls",
            color = Pitch.stump,
            fontSize = 32.sp,
            fontFamily = FontFamily.Serif,
            modifier = Modifier.padding(top = 18.dp, bottom = 8.dp),
        )
        if (balls.isEmpty()) {
            Text(
                text = "No balls yet. Film one, and it will show up here.",
                color = Pitch.stump,
                fontSize = 17.sp,
                modifier = Modifier.padding(top = 12.dp),
            )
            return
        }
        Column(Modifier.verticalScroll(rememberScrollState())) {
            balls.asReversed().forEach { ball ->
                BallRow(ball, onOpen)
                Box(Modifier.fillMaxWidth().height(1.dp).background(Pitch.stump.copy(alpha = 0.16f)))
            }
        }
    }
}

@Composable
private fun BallRow(ball: SavedBall, onOpen: (String) -> Unit) {
    val facts = remember(ball.sidecar.lastModified(), ball.sidecar.length()) {
        if (ball.sidecar.exists()) ballFacts(ball.sidecar.readText()) else null
    }
    val number = ballNumber(ball.id)
    Row(
        Modifier
            .fillMaxWidth()
            .clickable { onOpen(ball.id) }
            .padding(vertical = 16.dp)
            .semantics { contentDescription = "Open ball $number" },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(
                text = number,
                color = Pitch.stump,
                fontSize = 28.sp,
                fontFamily = FontFamily.Serif,
            )
            Text(
                text = rowDetail(facts),
                color = Pitch.stump.copy(alpha = 0.72f),
                fontSize = 15.sp,
            )
        }
        Text(
            text = facts?.takeIf { it.hasClock() }?.measuredFps?.let { String.format(Locale.US, "%.0f", it) } ?: "—",
            color = Pitch.far,
            fontSize = 28.sp,
            fontFamily = FontFamily.Serif,
            textAlign = TextAlign.End,
        )
    }
}

@Composable
fun BallDetail(
    ball: SavedBall?,
    onBack: () -> Unit,
    onDelete: (String) -> Unit,
) {
    val context = LocalContext.current
    var confirming by remember(ball?.id) { mutableStateOf(false) }
    val facts = remember(ball?.id, ball?.sidecar?.lastModified()) {
        ball?.sidecar?.takeIf { it.exists() }?.let { ballFacts(it.readText()) }
    }
    Column(
        Modifier
            .fillMaxSize()
            .windowInsetsPadding(androidx.compose.foundation.layout.WindowInsets.statusBars)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp),
    ) {
        Text(
            text = "Balls",
            color = Pitch.far,
            fontSize = 16.sp,
            modifier = Modifier
                .clickable(onClick = onBack)
                .padding(top = 16.dp, bottom = 8.dp)
                .semantics { contentDescription = "Back to balls" },
        )
        if (ball == null) {
            Text(text = "That ball is already gone.", color = Pitch.stump, fontSize = 20.sp)
            return
        }
        val number = ballNumber(ball.id)
        Text(
            text = "Ball $number",
            color = Pitch.stump,
            fontSize = 32.sp,
            fontFamily = FontFamily.Serif,
        )
        Spacer(Modifier.height(12.dp))
        ClipPlayer(ball, facts)
        Spacer(Modifier.height(18.dp))
        if (facts == null) {
            Text(
                text = "This clip has no JSON, so it cannot be tracked.",
                color = Pitch.stump,
                fontSize = 17.sp,
            )
        } else {
            Fact(
                "Frames a second",
                if (facts.hasClock()) {
                    facts.measuredFps?.let { String.format(Locale.US, "%.0f", it) } ?: "Missing"
                } else {
                    "Not in this file"
                },
            )
            Fact("Length", facts.durationSeconds?.let { String.format(Locale.US, "%.2f seconds", it) } ?: "Missing")
            Fact("Frames", facts.frameCount?.toString() ?: "Missing")
            Fact("Longest gap", facts.maxGapSeconds?.let { String.format(Locale.US, "%.0f ms", it * 1000) } ?: "Missing")
            Fact("Shutter", facts.exposureNsMedian?.let { String.format(Locale.US, "%.1f ms", it / 1_000_000.0) } ?: "Missing")
            Fact("ISO", facts.isoMedian?.toString() ?: "Missing")
            Fact("Lens", facts.focalLengthMm?.let { String.format(Locale.US, "%.2f mm", it) } ?: "Missing")
            Fact("Picture", pictureSize(facts))
            Fact("File", fileSize(ball.video.length()))
        }
        Spacer(Modifier.height(22.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            CreamButton("Send") { shareBall(context, ball) }
            if (!confirming) {
                Text(
                    text = "Delete",
                    color = Pitch.cherry,
                    fontSize = 17.sp,
                    modifier = Modifier
                        .clickable { confirming = true }
                        .padding(horizontal = 8.dp, vertical = 14.dp)
                        .semantics { contentDescription = "Delete ball $number" },
                )
            }
        }
        if (confirming) {
            Text(
                text = "Delete ball $number? The video and the JSON both go.",
                color = Pitch.stump,
                fontSize = 16.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.padding(top = 8.dp)) {
                Text(
                    text = "Delete ball $number",
                    color = Pitch.cherry,
                    fontSize = 17.sp,
                    modifier = Modifier
                        .clickable { onDelete(ball.id) }
                        .padding(vertical = 12.dp)
                        .semantics { contentDescription = "Confirm delete ball $number" },
                )
                Text(
                    text = "Keep it",
                    color = Pitch.stump,
                    fontSize = 17.sp,
                    modifier = Modifier
                        .clickable { confirming = false }
                        .padding(vertical = 12.dp),
                )
            }
        }
        Spacer(Modifier.height(28.dp))
    }
}

@Composable
private fun ClipPlayer(ball: SavedBall, facts: BallFacts?) {
    var playing by remember(ball.id) { mutableStateOf(false) }
    val wide = facts?.orientationHintDegrees == 0 || facts?.orientationHintDegrees == 180
    val ratio = if (wide) 16f / 9f else 9f / 16f
    Box(
        Modifier
            .fillMaxWidth()
            .height(280.dp)
            .clip(RoundedCornerShape(2.dp))
            .background(Pitch.ink)
            .clickable { playing = !playing }
            .semantics { contentDescription = if (playing) "Pause ball" else "Play ball" },
        contentAlignment = Alignment.Center,
    ) {
        AndroidView(
            factory = { context ->
                VideoView(context).apply {
                    setVideoURI(Uri.fromFile(ball.video))
                    setOnPreparedListener { player -> player.isLooping = true }
                }
            },
            update = { view ->
                if (playing && !view.isPlaying) view.start()
                if (!playing && view.isPlaying) view.pause()
            },
            onRelease = { view ->
                view.stopPlayback()
            },
            modifier = Modifier.fillMaxHeight().aspectRatio(ratio),
        )
        if (!playing) {
            Text(
                text = "Play",
                color = Pitch.ink,
                fontSize = 16.sp,
                modifier = Modifier
                    .align(Alignment.Center)
                    .clip(RoundedCornerShape(20.dp))
                    .background(Pitch.stump)
                    .padding(horizontal = 18.dp, vertical = 8.dp),
            )
        }
    }
}

@Composable
private fun Fact(name: String, value: String) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 7.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(text = name, color = Pitch.stump.copy(alpha = 0.72f), fontSize = 16.sp)
        Text(text = value, color = Pitch.stump, fontSize = 16.sp, fontFamily = FontFamily.Serif)
    }
}

@Composable
private fun CreamButton(label: String, onClick: () -> Unit) {
    Text(
        text = label,
        color = Pitch.ink,
        fontSize = 16.sp,
        modifier = Modifier
            .clip(RoundedCornerShape(24.dp))
            .background(Pitch.stump)
            .clickable(onClick = onClick)
            .padding(horizontal = 22.dp, vertical = 12.dp),
    )
}

private fun rowDetail(facts: BallFacts?): String {
    if (facts == null) return "No JSON with this video"
    if (!facts.hasClock()) return "No frame times in the JSON"
    return String.format(Locale.US, "%.1f seconds", facts.durationSeconds)
}

private fun BallFacts.hasClock(): Boolean = (frameCount ?: 0) > 0 && (durationSeconds ?: 0.0) > 0.0

private fun pictureSize(facts: BallFacts): String {
    val w = facts.encodedWidth
    val h = facts.encodedHeight
    if (w == null || h == null) return "Missing"
    val turned = facts.orientationHintDegrees == 90 || facts.orientationHintDegrees == 270
    return if (turned) "$h × $w" else "$w × $h"
}

private fun fileSize(bytes: Long): String {
    val mb = bytes / (1024.0 * 1024.0)
    return if (mb >= 1.0) String.format(Locale.US, "%.1f MB", mb) else "${bytes / 1024} KB"
}

fun ballNumber(id: String): String = id.removePrefix("ball-").trimStart('0').ifEmpty { "0" }

fun shareBall(context: android.content.Context, ball: SavedBall) {
    val files = listOf(ball.video, ball.sidecar, ball.frame).filter { it.exists() }
    if (files.isEmpty()) return
    val authority = context.packageName + ".files"
    val uris = files.map { FileProvider.getUriForFile(context, authority, it) }
    val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "*/*"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        clipData = ClipData.newRawUri("ball", uris.first()).also { clip ->
            uris.drop(1).forEach { clip.addItem(ClipData.Item(it)) }
        }
    }
    context.startActivity(Intent.createChooser(intent, "Send this ball"))
}
