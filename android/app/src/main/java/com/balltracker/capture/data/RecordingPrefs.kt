package com.balltracker.capture.data

import android.content.Context

data class RecordingOptions(
    val maxClipSeconds: Int = 45,
    val keepPreview: Boolean = false,
)

class RecordingPrefs(context: Context) {
    private val prefs = context.getSharedPreferences("recording", Context.MODE_PRIVATE)

    fun load(): RecordingOptions = RecordingOptions(
        maxClipSeconds = clipLimitSeconds(prefs.getInt(KEY_LIMIT, 45)),
        keepPreview = prefs.getBoolean(KEY_PREVIEW, false),
    )

    fun save(options: RecordingOptions) {
        prefs.edit()
            .putInt(KEY_LIMIT, clipLimitSeconds(options.maxClipSeconds))
            .putBoolean(KEY_PREVIEW, options.keepPreview)
            .apply()
    }

    companion object {
        private const val KEY_LIMIT = "maxClipSeconds"
        private const val KEY_PREVIEW = "keepPreview"
    }
}
