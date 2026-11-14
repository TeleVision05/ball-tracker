package com.balltracker.capture.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.balltracker.capture.data.RecordingOptions

@Composable
fun SettingsScreen(
    options: RecordingOptions,
    onClipSeconds: (Int) -> Unit,
    onKeepPreview: (Boolean) -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .windowInsetsPadding(androidx.compose.foundation.layout.WindowInsets.statusBars)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp),
    ) {
        Text(
            text = "Settings",
            color = Pitch.stump,
            fontSize = 32.sp,
            fontFamily = FontFamily.Serif,
            modifier = Modifier.padding(top = 18.dp, bottom = 18.dp),
        )
        Text(text = "Clip length", color = Pitch.stump, fontSize = 18.sp)
        Text(
            text = "Recording stops on its own at this time.",
            color = Pitch.stump.copy(alpha = 0.72f),
            fontSize = 15.sp,
            modifier = Modifier.padding(top = 4.dp, bottom = 12.dp),
        )
        Row {
            listOf(15, 30, 45).forEach { seconds ->
                Choice(
                    label = "${seconds}s",
                    selected = options.maxClipSeconds == seconds,
                    description = "Stop after $seconds seconds",
                    onClick = { onClipSeconds(seconds) },
                )
            }
        }
        Spacer(Modifier.height(28.dp))
        Text(text = "Picture while filming", color = Pitch.stump, fontSize = 18.sp)
        Spacer(Modifier.height(10.dp))
        PreviewChoice(
            title = "Hold still",
            body = "The picture freezes and the file keeps more frames. Watch the timer.",
            selected = !options.keepPreview,
            description = "Hold the picture still while filming",
            onClick = { onKeepPreview(false) },
        )
        Spacer(Modifier.height(8.dp))
        PreviewChoice(
            title = "Keep moving",
            body = "You can watch the ball. The file may miss more frames.",
            selected = options.keepPreview,
            description = "Keep the picture moving while filming",
            onClick = { onKeepPreview(true) },
        )
        Spacer(Modifier.height(28.dp))
    }
}

@Composable
private fun Choice(label: String, selected: Boolean, description: String, onClick: () -> Unit) {
    Text(
        text = label,
        color = if (selected) Pitch.ink else Pitch.stump,
        fontFamily = FontFamily.Serif,
        fontSize = 18.sp,
        modifier = Modifier
            .padding(end = 8.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(if (selected) Pitch.stump else Pitch.stump.copy(alpha = 0.12f))
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp)
            .semantics { contentDescription = description },
    )
}

@Composable
private fun PreviewChoice(
    title: String,
    body: String,
    selected: Boolean,
    description: String,
    onClick: () -> Unit,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(2.dp))
            .background(if (selected) Pitch.stump.copy(alpha = 0.14f) else Pitch.stump.copy(alpha = 0.05f))
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 12.dp)
            .semantics { contentDescription = description },
    ) {
        Text(text = title, color = if (selected) Pitch.far else Pitch.stump, fontSize = 17.sp)
        Text(
            text = body,
            color = Pitch.stump.copy(alpha = 0.8f),
            fontSize = 15.sp,
            modifier = Modifier.padding(top = 4.dp),
        )
    }
}
