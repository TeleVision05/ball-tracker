package com.balltracker.capture.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

object Pitch {
    val ground = Color(0xFF141814)
    val stump = Color(0xFFF3EFE4)
    val far = Color(0xFF8FCBB8)
    val cherry = Color(0xFFD64532)
    val ink = Color(0xFF141814)
}

private val scheme = darkColorScheme(
    primary = Pitch.stump,
    onPrimary = Pitch.ink,
    background = Pitch.ground,
    surface = Pitch.ground,
    error = Pitch.cherry,
)

private val type = Typography(
    bodyLarge = TextStyle(fontSize = 20.sp, lineHeight = 26.sp, fontWeight = FontWeight.Medium),
    labelLarge = TextStyle(fontSize = 16.sp, lineHeight = 20.sp, fontWeight = FontWeight.SemiBold),
)

@Composable
fun BallTrackerTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = scheme, typography = type, content = content)
}
